# instructzero_optimized.py
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import functools
import random

# Optional imports (fallback safely if unavailable)
try:
    import torch
    from torch import nn
except ImportError:
    torch = None

try:
    import botorch
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    botorch = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# -----------------------------------------------------------------------------
# 1. Simple profiling context manager
@contextmanager
def profile_block(name, logger=None):
    start = time.time()
    yield
    end = time.time()
    duration = end - start
    msg = f"[PROFILE] {name}: {duration:.3f}s"
    if logger:
        try:
            logger.info(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

# -----------------------------------------------------------------------------
# 2. Efficient kernel with cached Cholesky decomposition (avoids inverse)
class EfficientCombinedStringKernel:
    def __init__(self, instruction_kernel_obj, latent_space_kernel_obj, jitter=1e-4):
        """
        instruction_kernel_obj and latent_space_kernel_obj must expose a .forward(...) method
        that returns their kernel matrices on training data. This class caches their Cholesky
        factorizations and uses cholesky_solve instead of inverse.
        """
        if torch is None:
            raise RuntimeError("Torch is required for EfficientCombinedStringKernel.")
        self.jitter = jitter
        self.instruction_kernel = instruction_kernel_obj
        self.latent_space_kernel = latent_space_kernel_obj
        self._prepare_fixed()

    def _prepare_fixed(self):
        # Precompute and cache decompositions for fixed training kernels
        with torch.no_grad():
            K_latent = self.latent_space_kernel.forward()  # adapt signature if needed
            K_latent = K_latent + self.jitter * torch.eye(K_latent.size(0), device=K_latent.device)
            self.latent_L = torch.linalg.cholesky(K_latent)

            K_instr = self.instruction_kernel.forward()  # adapt signature if needed
            K_instr = K_instr + self.jitter * torch.eye(K_instr.size(0), device=K_instr.device)
            self.instr_L = torch.linalg.cholesky(K_instr)
            # store base instruction kernel if needed downstream
            self.K_instr = K_instr

    def _solve(self, L, B):
        # Solve A X = B where A = L L^T
        return torch.cholesky_solve(B, L)

    def forward(self, K_z1_training, K_z2_training, K_train_instruction):
        """
        Reconstruct combined kernel value efficiently. Matches the structure in original code but avoids
        explicit inverse.
        """
        # A^{-1} K_train_instruction
        # 复用缓存，避免重复算
        K_train_instruction = self.K_instr

        # 左右各打一遍 A^{-1}
        left_T = torch.linalg.solve_triangular(self.latent_L, K_z1_training.T, upper=False)  # L^{-1} K^T
        left = torch.linalg.solve_triangular(self.latent_L.T, left_T, upper=True).T  # (K A^{-1})

        right_T = torch.linalg.solve_triangular(self.latent_L, K_z2_training.T, upper=False)
        right = torch.linalg.solve_triangular(self.latent_L.T, right_T, upper=True).T

        kernel_val = left @ K_train_instruction @ right.T

        left = self._solve(self.latent_L, K_z1_training.T).T
        right = self._solve(self.latent_L, K_z2_training.T).T
        kernel_val = left @ middle @ right.T
        return kernel_val

# -----------------------------------------------------------------------------
# 3. Tokenization caching for repeated static prefix / system prompts
class TokenizerCache:
    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.cache = {}
        self.lock = threading.Lock()
        self.device = device

    def encode(self, prompt):
        key = prompt
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        if self.device:
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with self.lock:
            self.cache[key] = tokenized
        return tokenized

# -----------------------------------------------------------------------------
# 4. Batched / parallel LLM evaluation wrapper
class BatchLLMCaller:
    def __init__(self, model_forward_api, max_workers=1):
        """
        Wraps an existing model_forward_api which exposes .eval (and optionally .eval_async).
        Provides parallel and batched evaluation.
        """
        self.api = model_forward_api
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def eval_batch(self, X):
        """
        Synchronous parallel evaluation of a list of inputs (preserves order).
        """
        futures = [self.executor.submit(self.api.eval, x) for x in X]
        results = [None] * len(X)
        for idx, fut in enumerate(futures):
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"error": str(e)}
        return results

    async def eval_batch_async(self, X):
        """
        If the wrapped api has an async version use it; else fallback to sync parallel.
        """
        if hasattr(self.api, "eval_async"):
            import asyncio
            tasks = [self.api.eval_async(x) for x in X]
            return await asyncio.gather(*tasks)
        else:
            return self.eval_batch(X)

# -----------------------------------------------------------------------------
# 5. Acquisition optimization helper (multi-start) with fallback if botorch is unavailable
def optimize_acquisition_function(acq_func, bounds, q=1, num_restarts=5, raw_samples=128, device=None):
    """
    Wrapper to optimize an acquisition function. If botorch is present uses its optimize_acqf.
    Otherwise performs random sampling fallback.
    Returns (best_candidate, best_acq_value)
    """
    if botorch is not None:
        try:
            candidates, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            return candidates.detach(), acq_value.detach()
        except Exception as e:
            print(f"[WARNING] botorch optimize_acqf failed ({e}); falling back to random search.")

    # Fallback: random search over bounds
    if torch is None:
        raise RuntimeError("Torch is required for fallback acquisition optimization.")
    low = bounds[0]
    high = bounds[1]
    dim = low.shape[0]
    best_val = None
    best_pt = None
    for _ in range(raw_samples):
        candidate = low + (high - low) * torch.rand(dim, device=device)
        val = acq_func(candidate.unsqueeze(0))
        if best_val is None or val.item() > best_val:
            best_val = val.item()
            best_pt = candidate
    return best_pt.unsqueeze(0), torch.tensor([best_val], device=device)

# -----------------------------------------------------------------------------
# 6. Structured logging / telemetry
class TrainerLogger:
    def __init__(self, log_dir="runs/instructzero", use_tensorboard=True):
        self.use_tb = use_tensorboard and SummaryWriter is not None
        self.writer = SummaryWriter(log_dir) if self.use_tb else None
        self.best_so_far = None
        import logging
        self.logger = logging.getLogger("InstructZero")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def log_iteration(self, iteration, best_value, gp_loss=None, extra=None):
        msg = f"Iter {iteration} best_value={best_value:.5f}"
        if gp_loss is not None:
            msg += f" gp_loss={gp_loss:.5f}"
        if extra:
            msg += f" extra={extra}"
        self.logger.info(msg)

        if self.use_tb:
            self.writer.add_scalar("best_value", best_value, iteration)
            if gp_loss is not None:
                self.writer.add_scalar("gp_loss", gp_loss, iteration)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    try:
                        self.writer.add_scalar(k, v, iteration)
                    except Exception:
                        pass

        if self.best_so_far is None or best_value > self.best_so_far:
            self.best_so_far = best_value

    def close(self):
        if self.use_tb and self.writer:
            self.writer.close()

# -----------------------------------------------------------------------------
# 7. Retry decorator with exponential backoff for unstable calls (e.g., API)
def retry_with_backoff(max_attempts=3, initial_delay=0.5, factor=2.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    sleep_time = delay + random.random() * 0.1
                    time.sleep(sleep_time)
                    delay *= factor
        return wrapped
    return decorator
