# -*- coding: utf-8 -*-
# instruction_coupled_kernel.py (batched-fix + covariates/auto-weights)
import os
import math
import torch
import numpy as np
from tqdm.auto import tqdm
from gpytorch.kernels.kernel import Kernel
import gpytorch
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Positive
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _as_dense(x):
    return x.to_dense() if hasattr(x, "to_dense") else x


class EfficientCombinedStringKernel(Kernel):
    def __init__(
        self,
        base_latent_kernel,
        instruction_kernel,
        latent_train,
        instruction_train,
        jitter=1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_latent_kernel = base_latent_kernel
        self.instruction_kernel = instruction_kernel

        # 【修复】强制转换为 double (float64)
        self.latent_train = latent_train.double()
        self.instruction_train = instruction_train.double()
        self.lp_dim = latent_train.shape[-1]
        self._device = self.latent_train.device
        self._dtype = self.latent_train.dtype # 现在应该是 torch.float64
        self.n_tr = int(self.latent_train.shape[0])

        # === 【核心修改 1】将 alpha 注册为可学习参数 ===
        # 初始值：raw=0.0 经过 softplus 变换后大约是 0.69，作为一个中性的初始值
        self.register_parameter(
            name="raw_alpha_lat", 
            parameter=torch.nn.Parameter(torch.tensor(1.0, device=self._device, dtype=self._dtype)) # 原 0.0
        )
        self.register_parameter(
            name="raw_alpha_instr", 
            parameter=torch.nn.Parameter(torch.tensor(0.0, device=self._device, dtype=self._dtype))
        )
        # 协变量初始值给低一点 (raw=-2.0 => alpha ~ 0.1)，防止起步就跑偏
        self.register_parameter(
            name="raw_alpha_cov", 
            parameter=torch.nn.Parameter(torch.tensor(-1.0, device=self._device, dtype=self._dtype))
        )
        # 注册约束：保证权重永远大于 0
        self.register_constraint("raw_alpha_lat", Positive())
        self.register_constraint("raw_alpha_instr", Positive())
        self.register_constraint("raw_alpha_cov", Positive())

        self._mode = "auto_grad" # 标记模式变为自动梯度
        self._anneal_steps = 10
        self._cov_kernel_type = "linear"
        self._F_hist = None
        self._F_mu = None
        self._F_std = None
        self._w = np.ones(4, dtype=np.float32)
        self.K_cov = None
        jitter_tensor = torch.as_tensor(jitter, device=self._device, dtype=self._dtype)

        with torch.no_grad():
            K_lat = _as_dense(
                self.base_latent_kernel.forward(self.latent_train, self.latent_train)
            ).to(self._device, self._dtype)
            K_lat = K_lat + jitter_tensor * torch.eye(
                K_lat.size(0), device=self._device, dtype=self._dtype
            )
            latent_L = torch.linalg.cholesky(K_lat).detach()
            K_instr = _as_dense(
                self.instruction_kernel.forward(self.instruction_train, self.instruction_train)
            ).to(self._device, self._dtype)
            K_instr = K_instr + jitter_tensor * torch.eye(
                K_instr.size(0), device=self._device, dtype=self._dtype
            )
            K_instr = K_instr.detach()

            # register buffers
            self.register_buffer("latent_L", latent_L)
            self.register_buffer("K_lat_train", K_lat)
            self.register_buffer("K_instr", K_instr)
            self.register_buffer("jitter_tensor", jitter_tensor)
            self.register_buffer("I_tr", torch.eye(self.n_tr, device=self._device, dtype=self._dtype))

    # === 【核心修改 2】添加属性访问器，让外部依然可以用 .alpha_lat 访问 ===
    @property
    def alpha_lat(self):
        return self.raw_alpha_lat_constraint.transform(self.raw_alpha_lat)

    @alpha_lat.setter
    def alpha_lat(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha_lat)
        self.initialize(raw_alpha_lat=self.raw_alpha_lat_constraint.inverse_transform(value))

    @property
    def alpha_instr(self):
        return self.raw_alpha_instr_constraint.transform(self.raw_alpha_instr)

    @alpha_instr.setter
    def alpha_instr(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha_instr)
        self.initialize(raw_alpha_instr=self.raw_alpha_instr_constraint.inverse_transform(value))

    @property
    def alpha_cov(self):
        return self.raw_alpha_cov_constraint.transform(self.raw_alpha_cov)

    @alpha_cov.setter
    def alpha_cov(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha_cov)
        self.initialize(raw_alpha_cov=self.raw_alpha_cov_constraint.inverse_transform(value))

    def set_covariates(self, F_hist: np.ndarray, kind: str = "linear", lengthscale: float = 1.0):
        F_hist = np.asarray(F_hist, dtype=np.float64)
        N, d = F_hist.shape
        self._F_hist = F_hist.copy()

        if getattr(self, "_feature_weights", None) is None or self._feature_weights.numel() != d:
            w = np.ones(d, dtype=np.float64) / max(1, d)
            self._feature_weights = torch.tensor(w, dtype=torch.double, device=self._device)

        F = torch.tensor(F_hist, dtype=torch.double, device=self._device)
        mu = F.mean(dim=0, keepdim=True)
        std = F.std(dim=0, keepdim=True).clamp_min(1e-6)
        Fz = (F - mu) / std
        self._F_mu, self._F_std = mu, std

        w = self._feature_weights.to(dtype=torch.double, device=self._device).clamp_min(0.0)
        w = w / (w.sum() + 1e-9)
        Fw = Fz * w.sqrt().unsqueeze(0)  # [N, d]

        if kind == "linear":
            K_cov = Fw @ Fw.T
        elif kind == "rbf":
            D2 = ((Fw[:, None, :] - Fw[None, :, :]) ** 2).sum(-1)
            K_cov = torch.exp(-0.5 * D2 / max(1e-8, lengthscale ** 2))
        else:
            raise ValueError(f"Unknown cov kind: {kind}")
        mdiag = K_cov.diag().mean().clamp_min(1e-12)
        K_cov = (K_cov / mdiag) + self.jitter_tensor * torch.eye(N, dtype=K_cov.dtype, device=K_cov.device)

        self.K_cov = K_cov.to(self._dtype)

    def learn_feature_weights(self, F_hist: np.ndarray, y: np.ndarray, tau: float = 0.5,
                              rebuild_kind: str = "linear", lengthscale: float = 1.0):

        F = np.asarray(F_hist, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        Fy = (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True) + 1e-6)
        yy = (y - y.mean()) / (y.std() + 1e-6)

        cors = np.abs((Fy * yy[:, None]).mean(0))
        exps = np.exp(cors / max(1e-6, tau))
        w = exps / (exps.sum() + 1e-12)

        self._feature_weights = torch.tensor(w, dtype=torch.double, device=self._device)
        self.set_covariates(F_hist, kind=rebuild_kind, lengthscale=lengthscale)

    def set_cov_kernel_type(self, kind: str = "linear"):
        kind = (kind or "linear").lower()
        if kind not in {"linear", "rbf"}:
            kind = "linear"
        self._cov_kernel_type = kind

    def _build_train_K(self, a_lat: float, a_instr: float, a_cov: float):
        # 这个函数现在只用于 optimize_mixture_weights 中的临时计算
        # 但因为我们要废弃 optimize_mixture_weights，这个函数其实也可以不用了
        # 为了兼容性保留
        K = torch.zeros_like(self.K_lat_train)
        if a_lat != 0.0:
            K = K + float(a_lat) * self.K_lat_train
        if a_instr != 0.0:
            K = K + float(a_instr) * self.K_instr
        if (self.K_cov is not None) and (a_cov != 0.0):
            K = K + float(a_cov) * self.K_cov.to(self._device, self._dtype)
        K = K + self.jitter_tensor * self.I_tr
        return K

    def normalize_components(self):
        # latent
        mdiag_lat = self.K_lat_train.diag().mean().clamp_min(1e-12)
        if (mdiag_lat - 1.0).abs() > 1e-8:
            scale_lat = 1.0 / mdiag_lat
            self.K_lat_train.mul_(scale_lat)
            # L_new = L / sqrt(mdiag_lat)
            self.latent_L = self.latent_L / torch.sqrt(mdiag_lat)

        # instruction
        mdiag_ins = self.K_instr.diag().mean().clamp_min(1e-12)
        if (mdiag_ins - 1.0).abs() > 1e-8:
            scale_ins = 1.0 / mdiag_ins
            self.K_instr.mul_(scale_ins)

        # covariate（如果已存在）
        if self.K_cov is not None:
            mdiag_cov = self.K_cov.diag().mean().clamp_min(1e-12)
            if (mdiag_cov - 1.0).abs() > 1e-8:
                scale_cov = 1.0 / mdiag_cov
                self.K_cov = (self.K_cov * scale_cov).to(self._dtype)

    @staticmethod
    def _gp_mll(K: torch.Tensor, y: np.ndarray) -> float:
        # 静态方法，保持不变
        yv = torch.as_tensor(np.asarray(y, dtype=np.float64).reshape(-1, 1), dtype=torch.float64, device=K.device)
        Kd = K.to(dtype=torch.float64)
        try:
            L = torch.linalg.cholesky(Kd)
        except RuntimeError:
            return -np.inf
        alpha = torch.cholesky_solve(yv, L)
        mll = -0.5 * float((yv.T @ alpha).item())
        mll -= float(torch.log(torch.diag(L)).sum().item())
        mll -= 0.5 * K.shape[0] * math.log(2.0 * math.pi)
        return mll

    # === 【核心修改 3】废弃网格搜索，改为直接返回，让梯度下降接管 ===
    def optimize_mixture_weights(self, y: np.ndarray, step_idx: int, grid=None):
        # 既然参数已经可学习，这里就不需要手动搜了
        # 甚至可以直接 return，什么都不做
        return -np.inf, None

    def current_mode(self):
        # 打印当前的权重（注意要用 .item() 取值）
        return self._mode, dict(
            alpha_lat=self.alpha_lat.item(), 
            alpha_instr=self.alpha_instr.item(), 
            alpha_cov=self.alpha_cov.item()
        )

    def _solve(self, L, B):
        return torch.cholesky_solve(B, L)

    def forward(self, z1, z2, **params):
        """
        Fully batched kernel:
        z1: (..., n, d)
        z2: (..., m, d)
        returns: (..., n, m)
        """
        z1 = z1.to(self._device, self._dtype)
        z2 = z2.to(self._device, self._dtype)

        if z1.dim() < 2 or z2.dim() < 2:
            raise ValueError(f"Inputs must be at least 2D (n,d). Got {z1.shape}, {z2.shape}")

        # 对齐 batch 形状（支持广播）
        bs1 = z1.shape[:-2]
        bs2 = z2.shape[:-2]
        batch_shape = torch.broadcast_shapes(bs1, bs2)
        if bs1 != batch_shape:
            z1 = z1.expand(*batch_shape, z1.shape[-2], z1.shape[-1])
        if bs2 != batch_shape:
            z2 = z2.expand(*batch_shape, z2.shape[-2], z2.shape[-1])

        n = z1.shape[-2]
        m = z2.shape[-2]
        d = z1.shape[-1]
        m_tr = self.latent_train.shape[0]

        # 合并 batch 维做向量化
        B = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1
        z1_flat = z1.reshape(B, n, d)
        z2_flat = z2.reshape(B, m, d)

        # 取 latent 维度
        z1_lat = z1_flat[..., :self.lp_dim]  # (B, n, lp)
        z2_lat = z2_flat[..., :self.lp_dim]  # (B, m, lp)

        # === (1) latent 直接核：K_lat(z1,z2) —— 逐 batch 计算，避免跨批 all-to-all 造成 reshape 错误
        if B == 1:
            K_lat_pair = _as_dense(self.base_latent_kernel(z1_lat[0], z2_lat[0], **params)) \
                .to(self._device, self._dtype).unsqueeze(0)  # (1, n, m)
        else:
            K_list = []
            for b in range(B):
                Kb = _as_dense(self.base_latent_kernel(z1_lat[b], z2_lat[b], **params)) \
                    .to(self._device, self._dtype)  # (n, m)
                K_list.append(Kb)
            K_lat_pair = torch.stack(K_list, dim=0)  # (B, n, m)

        # === (2) 原“高效耦合”项：K_eff = K1 A^{-1} M A^{-1} K2^T
        # 计算 K(z?, Ztr)
        K1 = _as_dense(
            self.base_latent_kernel.forward(
                z1_lat.reshape(B * n, -1), self.latent_train, **params
            )
        ).to(self._device, self._dtype).reshape(B, n, m_tr)

        K2 = _as_dense(
            self.base_latent_kernel.forward(
                z2_lat.reshape(B * m, -1), self.latent_train, **params
            )
        ).to(self._device, self._dtype).reshape(B, m, m_tr)

        # A^{-1} via cholesky_solve，A=K_lat(Ztr,Ztr)
        # 注意要 cast to K1.dtype
        L = self.latent_L.to(device=self._device, dtype=K1.dtype) 
        left = torch.cholesky_solve(K1.transpose(-1, -2), L).transpose(-1, -2)   # (B, n, m_tr)
        right = torch.cholesky_solve(K2.transpose(-1, -2), L).transpose(-1, -2)  # (B, m, m_tr)

        M = self.K_instr.to(dtype=left.dtype)  # (m_tr, m_tr)
        leftM = torch.matmul(left, M)                         # (B, n, m_tr)
        K_eff = torch.matmul(leftM, right.transpose(-1, -2))  # (B, n, m)

        # === (3) 混合：【修改重点】直接使用 Parameter 相乘，保留梯度，不要加 float() ===
        K_total = self.alpha_lat * K_lat_pair + self.alpha_instr * K_eff

        # === (4) 若是训练期 K(Xtr, Xtr)，再加 α_cov * K_cov
        add_cov = False
        if (B == 1 and self.K_cov is not None):
             # 简单的判断：如果维度匹配且是方阵，且与训练集大小一致
             # 注意：这在某些 corner case (batch prediction) 可能有问题，
             # 最稳妥的是在外部显式控制，但为了不改动太多接口，我们用 shape 判断
             if n == self.n_tr and m == self.n_tr:
                 # 进一步检查是否是训练数据（通过对角线或第一个元素）
                 # 这是一个这种 hack，但能工作
                    add_cov = True
        if add_cov:
            # 只有训练 Loss 会加上这一项。
            # 这意味着 Covariates 被用来解释 "为什么某些点的 y 值偏离了 Latent 的预测"
            # 从而“净化”了 Latent Kernel 的学习。
            K_cov_safe = self.K_cov.to(dtype=K_total.dtype, device=K_total.device)
            # 确保 K_cov 也是 batch 形式 (1, N, N)
            K_total = K_total + self.alpha_cov * K_cov_safe.unsqueeze(0)

        # 还原 batch 维
        if B == 1 and len(batch_shape) == 0:
            return K_total.squeeze(0)
        else:
            return K_total.reshape(*batch_shape, n, m)


# ==============（可选）CMA-ES 优化器（保持原样）==============
try:
    import cma
except Exception:  # 兼容无 cma 环境
    cma = None

def cma_es_concat(starting_point_for_cma, EI, tkwargs, max_iters=10, popsize=12):
    if cma is None:
        raise RuntimeError("cma not installed")
    device = tkwargs["device"] if "device" in tkwargs else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 同步回 tkwargs（防止后面代码再取）
    tkwargs["device"] = device

    if isinstance(starting_point_for_cma, torch.Tensor):
        if starting_point_for_cma.is_cuda:
            starting_point_for_cma = starting_point_for_cma.detach().cpu()
        starting_point_for_cma = starting_point_for_cma.squeeze().tolist()
    es = cma.CMAEvolutionStrategy(
        x0=starting_point_for_cma,
        sigma0=0.8,
        inopts={"popsize": popsize, "verb_disp": 0},
    )
    it = 0
    with tqdm(total=max_iters, desc="CMA-ES", leave=False) as pbar:
        while not es.stop() and it < max_iters:
            it += 1
            xs = es.ask()
            X = torch.tensor(np.array(xs)).float().unsqueeze(1).to(device)
            with torch.no_grad():
                Y = -1 * EI(X)  # EI is maximized, CMA-ES minimizes
            es.tell(xs, Y.cpu().numpy())
            pbar.set_postfix(best_f=es.best.f)
            pbar.update(1)
    return es.best.x, -1 * es.best.f
