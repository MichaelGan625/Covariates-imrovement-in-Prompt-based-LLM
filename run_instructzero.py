import os
import re
import time
import copy
import random
import math
import csv
import semantic_covariates
from semantic_covariates import SemanticCovariateSystem
from datetime import datetime
from gpytorch.kernels import MaternKernel, LinearKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
import gpytorch
import torch
import numpy as np
import numpy as _np
import torch
import torch.nn.functional as F
import gpytorch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
import gpytorch
try:
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
except Exception:
    qLogNoisyExpectedImprovement = None
from tqdm.auto import tqdm
from gpytorch.constraints import GreaterThan  # 噪声下限约束
from botorch.exceptions import InputDataWarning
import warnings
warnings.filterwarnings("ignore", category=InputDataWarning)
# LLM/APE 相关
from automatic_prompt_engineer import ape, evaluate, config, template, data
from data_instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer

# 你自己项目里的工具
from instructzero_optimized import BatchLLMCaller, TrainerLogger, profile_block, optimize_acquisition_function
from misc import get_test_conf, get_conf, set_all_seed, TASKS, tkwargs, N_INIT, BATCH_SIZE, N_ITERATIONS
from args import parse_args
tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float64,  # 【关键修改】从 float32 改为 float64
}
# ==== [PATCH] 若你在 misc 里实现了 compute_four_covariates，这里优先使用；否则本文件有兜底 ====
try:
    from misc import compute_four_covariates as _compute_four_covariates_ext
except Exception:
    _compute_four_covariates_ext = None

# BO/Gaussian Process
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll

from gpytorch.kernels import ScaleKernel, MaternKernel, LinearKernel
from gpytorch.priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
ALLOW_SHORT_TASKS = {
    'antonyms', 'synonyms', 'diff', 'sum', 'negation', 
    'rhymes', 'singular_to_plural', 'active_to_passive', 
    'num_to_verbal', 'first_word_letter', 'second_word_letter',
    'letters_list', 'orthography_starts_with', 'taxonomy_animal',
    'larger_animal', 'word_sorting', 'word_unscrambling',
    'object_count', 'odd_one_out', 'ascii', 'periodic_elements',
    'translation_en-de', 'translation_en-es', 'translation_en-fr',
    'sentiment', 'sentence_similarity', 'common_concept'
}
# 内核与CMA-ES
from instruction_coupled_kernel import EfficientCombinedStringKernel
from instruction_coupled_kernel import *  # 含 cma_es_concat

# 任务→metric 的映射
from evaluation_instruction_induction import utility as _util

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --------------------- 小工具：查看开发集样例 ---------------------
def debug_dump_eval_samples(eval_data, k=8):
    ins, outs = eval_data
    print(f"[DEV SAMPLES] total={len(ins)} show_first={min(k, len(ins))}")
    for i in range(min(k, len(ins))):
        gold = outs[i]
        if isinstance(gold, (list, tuple)):
            gold_show = " | ".join(map(str, gold[:3]))
        else:
            gold_show = str(gold)
        print(f"[{i}] IN: {str(ins[i])[:140]}")
        print(f"    GOLD: {gold_show[:200]}")


# --------------------- 兼容 evaluate 返回 ---------------------
def _unpack_eval_return(ret):
    """
    兼容 eval_batch 的两种可能返回：
    - tuple/list: (y_scalar, y_scores_vector)
    - dict: 可能出现的键组合：
        ('dev_perf','dev_scores'), ('score','scores'), ('Y','S'), 或退化字段
    """
    import numpy as _np
    # tuple/list
    if isinstance(ret, (list, tuple)):
        if len(ret) == 0:
            return None, []
        y = ret[0] if len(ret) > 0 else None
        s = ret[1] if len(ret) > 1 else []
        return y, s

    # dict
    if isinstance(ret, dict):
        for yk, sk in [("dev_perf", "dev_scores"),
                       ("score", "scores"),
                       ("Y", "S"),
                       ("perf", "scores")]:
            if yk in ret:
                return ret.get(yk), ret.get(sk, [])
        # 兜底：取第一个数值当 y，第一个 list/ndarray 当 scores
        y = next((v for v in ret.values()
                  if isinstance(v, (int, float, _np.floating))), None)
        s = next((v for v in ret.values()
                  if isinstance(v, (list, tuple, _np.ndarray))), [])
        return y, s

    # 标量兜底
    return ret, []


# === Task profiles: 为避免“一刀切”，给部分任务单独策略（可按需增删）=== #
TASK_PROFILES = {
    # —— 强规则类：恢复“确定性+不清洗+Matern” —— #
    "first_word_letter":   {"deterministic": True, "clean": True, "kernel": "matern"},
"second_word_letter":  {"deterministic": True, "clean": True, "kernel": "matern"},
"letters_list":        {"deterministic": True, "clean": True, "kernel": "matern"},
"diff":                {"deterministic": True, "clean": True, "kernel": "matern"},
"sum":                 {"deterministic": True, "clean": True, "kernel": "matern"},
"num_to_verbal":       {"deterministic": True, "clean": True, "kernel": "matern"},
"singular_to_plural":  {"deterministic": True, "clean": True, "kernel": "matern"},
"orthography_starts_with": {"deterministic": True, "clean": True, "kernel": "matern"},


    # —— 需要多样性的：允许采样；指令核可用 cosine —— #
    "antonyms":            {"deterministic": False, "clean": True,  "kernel": "cosine"},
    "cause_and_effect":    {"deterministic": False, "clean": True,  "kernel": "cosine"},
}
DEFAULT_PROFILE = {"deterministic": False, "clean": True, "kernel": "matern"}

# “规则类任务”集合（协变量默认不开主导）
RULE_TASKS = {
    "first_word_letter", "second_word_letter", "letters_list",
    "diff", "sum", "num_to_verbal", "singular_to_plural",
    "orthography_starts_with"
}


# === baseline/simple GP helper for debugging and fallback ===
from botorch.models import SingleTaskGP as _SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood as _ExactMarginalLogLikelihood

def build_and_fit_simple_gp(X_train, y_train, device):
    target_dtype = tkwargs.get('dtype', torch.float64) 
    X_train = X_train.to(device)  
    y_train = y_train.to(device)

    # 诊断
    print(f"[baseline] X_train shape: {X_train.shape}, dtype: {X_train.dtype}, device: {X_train.device}")
    print(f"[baseline] y_train shape: {y_train.shape}, std: {y_train.std().item():.4e}, mean: {y_train.mean().item():.4e}")

    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=X_train.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0)
    ).to(device)
    covar_module = ScaleKernel(base_kernel=base_kernel).to(device)
    gp_model = _SingleTaskGP(X_train, y_train, covar_module=covar_module).to(device)
    mll = _ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    gp_model.train()
    try:
        with gpytorch.settings.cholesky_jitter(1e-2):
            fit_gpytorch_mll(mll)
    except Exception:
        import traceback
        print("=== baseline GP fit failed ===")
        traceback.print_exc()
        raise
    return gp_model, mll


def construct_custom_gp(X_train, y_train, latent_train, instruction_train, device, instr_kernel_type='matern'):
    # 1. 数据强制转 double
    target_dtype = tkwargs.get('dtype', torch.float64)
    Xd = X_train.to(device=device, dtype=target_dtype)
    yd = y_train.to(device=device, dtype=target_dtype)
    latent_train = latent_train.to(device=device, dtype=target_dtype)
    instruction_train = instruction_train.to(device=device, dtype=target_dtype)

    # 2. 基础核函数加 .double()
    base_latent = MaternKernel(
        nu=2.5,
        ard_num_dims=latent_train.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    ).to(device=device, dtype=target_dtype) # <--- 必须加 .double()

    if (instr_kernel_type or "").lower() == 'cosine':
        instruction_base = LinearKernel(ard_num_dims=instruction_train.shape[-1]).to(device).double()
    else:
        instruction_base = MaternKernel(
            nu=2.5,
            ard_num_dims=instruction_train.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ).to(device).double() # <--- 必须加 .double()

    # 3. 组合核加 .double()
    # 这一步最关键，因为报错的 latent_L 就在这里面
    combined_kernel = EfficientCombinedStringKernel(
        base_latent_kernel=base_latent,
        instruction_kernel=instruction_base,
        latent_train=latent_train, # 确保传入的是上面转过 double 的变量
        instruction_train=instruction_train,
        jitter=1e-2, 
    ).to(device).double() # <--- 必须加 .double()

    covar_module = ScaleKernel(base_kernel=combined_kernel).to(device).double()

    if yd.std() < 1e-6:
        outcome_transform = None
    else:
        outcome_transform = Standardize(m=1)

    # 4. GP 模型主体加 .double()
    gp_model = SingleTaskGP(
        Xd, yd,
        covar_module=covar_module,
        outcome_transform=outcome_transform,
    ).to(device).double() # <--- 必须加 .double()

    gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    return gp_model, gp_mll

def _cosine_bow(a: str, b: str) -> float:
    a = (a or "").lower()
    b = (b or "").lower()
    wa = re.findall(r"\w+", a)
    wb = re.findall(r"\w+", b)
    if not wa or not wb:
        return 0.0
    from collections import Counter
    va, vb = Counter(wa), Counter(wb)
    inter = set(va.keys()) & set(vb.keys())
    dot = sum(va[w] * vb[w] for w in inter)
    na = math.sqrt(sum(x * x for x in va.values()))
    nb = math.sqrt(sum(x * x for x in vb.values()))
    return 0.0 if na == 0 or nb == 0 else float(dot / (na * nb))



def _install_covariates_into_kernel(gp_model, F_hist_list, y_train_tensor, step_idx=0, allow_mix_opt=True):
    try:
        kern = gp_model.covar_module.base_kernel
    except Exception:
        return

    if not isinstance(F_hist_list, list) or len(F_hist_list) == 0:
        return

    try:
        F = _np.vstack(F_hist_list).astype(np.float64)
    except Exception:
        return
    try:

        if hasattr(kern, "set_covariates"):

            kern.set_covariates(F, kind="rbf", lengthscale=1.0)


        if hasattr(kern, "learn_feature_weights"):
            y_vec = y_train_tensor.detach().cpu().numpy().reshape(-1)
            kern.learn_feature_weights(F, y=y_vec, tau=0.5, rebuild_kind="rbf", lengthscale=1.0)


        if allow_mix_opt and hasattr(kern, "optimize_mixture_weights"):
            y_vec = y_train_tensor.detach().cpu().numpy().reshape(-1)
            kern.optimize_mixture_weights(y_vec, step_idx=step_idx)


        if hasattr(kern, "normalize_components"):
            kern.normalize_components()


        if hasattr(kern, "current_mode"):
            print("[kernel]", *kern.current_mode())

    except Exception as e:
        print("[kernel] covariate hooks failed (ignored):", e)


# --------------------- LMForwardAPI ---------------------
class LMForwardAPI:
    def __init__(self, model_name=None, eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None,
                 HF_cache_dir=None, args=None):
        # 先设设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ops_model = model_name
        if self.ops_model in ["vicuna", "wizardlm", 'openchat']:

            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir,
                low_cpu_mem_usage=True,
                device_map={"": "cuda:0"} if self.device.startswith("cuda") else None,
                local_files_only=True,
                torch_dtype=torch.float16,
                use_cache=True,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_cache_dir,
                model_max_length=1024,
                padding_side="left",
                use_fast=False,
                local_files_only=True,
            )

            # 静音上一轮的 temperature/top_p 警告：默认不采样
            gc = self.model.generation_config
            if hasattr(gc, "temperature"):
                gc.temperature = None
            if hasattr(gc, "top_p"):
                gc.top_p = None
            gc.do_sample = False
            self.model.generation_config = gc

        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if self.ops_model in ['wizardlm', 'vicuna', 'openchat']:
            self.embedding = self.model.get_input_embeddings().weight.to(self.device)
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.to(self.device)
            self.init_prompt = self.embedding[input_ids]

        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))

        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False).to(self.device)

        if self.ops_model == 'vicuna':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'wizardlm':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'alpaca':
            self.system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        else:
            NotImplementedError

        if random_proj == 'normal':
            if model_name in ['wizardlm', 'vicuna', 'openchat']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = self.embedding.reshape(-1).mean().item()
            std_hat = self.embedding.reshape(-1).std().item()
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            torch.nn.init.normal_(self.linear.weight, -1, 1)
        elif random_proj == 'uniform':
            torch.nn.init.uniform_(self.linear.weight, -1, 1)

        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        # 将 CLI 开关传入 evaluation 配置，供 exec_accuracy 使用
        try:
            self.conf.setdefault('evaluation', {})
            if getattr(args, 'print_examples', False):
                self.conf['evaluation']['print_examples'] = True
            if getattr(args, 'max_print', None) is not None:
                self.conf['evaluation']['max_print'] = int(args.max_print)
        except Exception:
            pass

        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput (result only): [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        self.api_model = args.api_model
        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        # 记录任务名供 profile 使用
        self.task_name = self.conf.get("evaluation", {}).get("task", None)

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.best_instruction = None
        self.num_call = 0
        self.prompts_set = dict()
        self.last_instruction_text = None  # 供主循环做去重

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)
        if isinstance(prompt_embedding, list):
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe, dtype=torch.float32, device=self.device)
                z = self.linear(z)
                pe_list.append(z)
            prompt_embedding = torch.cat(pe_list)
        elif isinstance(prompt_embedding, np.ndarray):
            prompt_embedding = torch.tensor(prompt_embedding, dtype=torch.float32, device=self.device)
            prompt_embedding = self.linear(prompt_embedding)
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor):
            prompt_embedding = prompt_embedding.type(torch.float32).to(self.device)
            prompt_embedding = self.linear(prompt_embedding)
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(f'[Prompt Embedding] Only support [list, numpy.ndarray], got {type(prompt_embedding)} instead.')

        input_text = self.init_token
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embed = torch.cat((prompt_embedding, input_embed), 1)

        # === 依据任务画像决定是否采样 ===
        _profile = TASK_PROFILES.get(self.task_name, DEFAULT_PROFILE)
        # === 1. 生成配置 (增加随机性，防止复读) ===
        # 这里把 temperature 稍微调高一点点，让它敢于生成不同的句子
        gen_kwargs = dict(do_sample=True, temperature=0.6, top_p=0.9,repetition_penalty=1.1)

        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=64, **gen_kwargs)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# === 2. 强力清洗 (Enhanced Cleaning) ===
        raw = instruction[0] if isinstance(instruction, list) else str(instruction)
        
        # 1. 去除常见的“前缀废话”
        # 使用正则表达式，忽略大小写，非贪婪匹配
        import re
        patterns = [
            r"^Instruction\s*:\s*", 
            r"^The instruction is\s*(to|:)?\s*",
            r"^The task is\s*(to|:)?\s*",
            r"^Based on.*?write\s*:\s*",
            r"^Task\s*:\s*"
        ]
        instr_clean = raw.strip()
        for pat in patterns:
            instr_clean = re.sub(pat, "", instr_clean, flags=re.IGNORECASE)
            
        # 2. 去除首尾的引号 (单引号和双引号)
        instr_clean = instr_clean.strip('"\'')
        
        # 3. 只要第一行 (防止模型生成指令后又开始解释)
        if "\n" in instr_clean:
            instr_clean = instr_clean.split("\n")[0].strip()
            
        # 4. (可选) 如果剩下的内容里还有 "Input:"，说明清洗失败或者是复读，强制截断
        if "Input:" in instr_clean:
            instr_clean = instr_clean.split("Input:")[0].strip()

        # 打印清洗结果，方便调试
        print(f'Raw: {raw[:50]}... -> Clean: {instr_clean}')

# === 3. 强力过滤器 (通用版) ===
        is_lazy_pattern = False
        lower_instr = instr_clean.lower()
        
        # 只拦截极其明显的“数据泄露”或“复读”行为
        # 例如：指令直接以 "Input:" 开头，说明它没写指令，而是在抄例子
        if instr_clean.strip().lower().startswith("input:"):
            is_lazy_pattern = True
            
        # 拦截模型拒绝回答的情况
        if "sorry" in lower_instr and "cannot" in lower_instr:
            is_lazy_pattern = True

        # [修改] 不再拦截包含 "output" 单词的指令，也不拦截短指令
        # 让 Llama-3 去打分，如果指令太短（如 "Sum"），Llama-3 也能执行，不应该判 0 分。
        
        if is_lazy_pattern:
            return 0.0, dummy_scores

        # === 4. 判决 ===
        if is_lazy_pattern:
            print(f"[FILTER] 拦截到偷懒指令: '{instr_clean}' -> 强制判为 0.0 分")
            # 记录一下，防止主循环报错找不到 last_instruction_text
            self.last_instruction_text = instr_clean 
            
            # 【核心修改】直接返回 0 分！给 GP 一个巨大的负反馈
            # 注意：这里的 [0.0] * len(...) 是为了构造一个全 0 的 scores 向量
            dummy_scores = [0.0] * len(self.eval_data[0])
            return 0.0, dummy_scores

        # === 5. 通过检查，正常进入后续评估 ===
        # 兜底：真的为空时（极少见），给 antonyms 一个最小可用指令
        if not instr_clean:
            if self.task_name == 'antonyms':
                instr_clean = "Return the antonym of the input word."

        instruction = [instr_clean]
        self.last_instruction_text = instr_clean  # 记录给主循环去重

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))

        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if self.api_model in [ 'meta-llama/llama-3-70b-instruct','meta-llama/llama-3-8b-instruct','meta-llama/llama-3.3-70b-instruct:free']:
                dev_perf, instruction_score = evaluate.evaluate_prompts(
                    instruction, self.eval_template, self.eval_data,
                    self.demos_template, self.few_shot_data,
                    self.conf['evaluation']['method'], self.conf['evaluation']
                )
                dev_perf = dev_perf.sorted()[1][0]
                # NaN/Inf 清洗
                try:
                    dev_perf = float(dev_perf)
                except Exception:
                    dev_perf = 0.0
                if not math.isfinite(dev_perf):
                    dev_perf = 0.0
                instruction_score = _np.nan_to_num(_np.asarray(instruction_score, dtype=float),
                                                   nan=0.0, posinf=0.0, neginf=0.0)
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score, instruction[0]

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set


# --------------------- BO 追踪 CSV ---------------------
def _init_trace(task):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join("logs", "bo_traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_{task}_{ts}.csv")
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "iter","phase","dev_perf_raw","dev_perf_for_gp",
            "instr_text","cov_sim","cov_delta","cov_negppl","cov_compliance",
            "alpha_lat","alpha_instr","alpha_cov",
            "acq","post_mu","post_sigma"
        ])
    return trace_path


def _append_trace(trace_path, iter_i, phase, dev_raw, dev_gp, instr_text, cov4, alphas, acq=None, mu=None, sigma=None):
    try:
        with open(trace_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            a_lat, a_instr, a_cov = (None, None, None)
            if isinstance(alphas, (tuple, list)) and len(alphas) == 3:
                a_lat, a_instr, a_cov = alphas
            sim, delta, negppl, comp = (None, None, None, None)
            if isinstance(cov4, (list, tuple, np.ndarray)) and len(cov4) == 4:
                sim, delta, negppl, comp = cov4
            w.writerow([
                iter_i, phase, dev_raw, dev_gp,
                (instr_text or "")[:180], sim, delta, negppl, comp,
                a_lat, a_instr, a_cov,
                acq, mu, sigma
            ])
    except Exception:
        pass


def _get_kernel_alphas(gp_model):
    try:
        kern = gp_model.covar_module.base_kernel
        mode, mix = kern.current_mode()
        return (mix.get("alpha_lat"), mix.get("alpha_instr"), mix.get("alpha_cov"))
    except Exception:
        return (None, None, None)


# --------------------- 主流程 ---------------------
def run(args):
    task, HF_cache_dir = args.task, args.HF_cache_dir
    random_proj, intrinsic_dim, n_prompt_tokens = args.random_proj, args.intrinsic_dim, args.n_prompt_tokens
    semantic_covariate_system = SemanticCovariateSystem(max_history=100)
    assert args.task in TASKS, 'Task not found!'
    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    if induce_data_size <= 20:
        # 数据极少时，对半分
        prompt_gen_size = int(induce_data_size * 0.5)
    else:
        # 数据充足时，最多取 50% 或 100条作为 Prompt池
        prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    
    # 兜底：如果切分后 eval_data 还是空的（极少见），强制复用
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)
    if len(eval_data[0]) == 0:
        print("[WARNING] Eval data is empty! Using induce data for both.")
        eval_data = induce_data

    # 可选：打印开发集样例（兼容额外 flag；没有就走 print_examples）
    if getattr(args, "print_eval_samples", getattr(args, "print_examples", False)):
        debug_dump_eval_samples(eval_data, k=int(getattr(args, "max_print", 8) or 8))

    # Data is in the form input: single item, output: list of items
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = (
        "Instruction: [PROMPT]\n\n"
        "Input: [INPUT]\n\n"
        "Constraint: Answer with the output only. No explanation.\n"
        "Output:"
    )
    init_prompt = ['\n']
    prompt_gen_template = (
        "I will provide some input-output pairs.\n\n"
        "[full_DEMO]\n\n"
        "The instruction that describes the relationship between the inputs and outputs is:\n"
        "Instruction:"
    )
    base_conf = '../configs/instruction_induction.yaml'
    conf = get_conf(task, eval_data)

    # ================= [最小修改修复 Start] =================
    # 获取池子里实际有多少条数据
    n_available = len(prompt_gen_data[0])
    # 获取配置文件想要多少个 demo (默认通常是 4 或 8)
    n_wanted = conf['generation'].get('num_demos', 4)

    # 【关键】取最小值，防止 ValueError: Sample larger than population
    safe_num_demos = min(n_available, n_wanted)

    # 简单的日志提示，让你知道发生了什么
    if safe_num_demos < n_wanted:
        print(f"[WARNING] Available data ({n_available}) < Config required ({n_wanted}). Using {safe_num_demos} demos.")

    # make the demo automatically
    # 这里把 conf['generation']['num_demos'] 替换为 safe_num_demos
    subsampled_data = data.subsample_data(prompt_gen_data, safe_num_demos)
    # ================= [最小修改修复 End] =================

    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]

    model_forward_api = LMForwardAPI(
        model_name=args.model_name,
        eval_data=eval_data,
        init_prompt=init_prompt,
        init_qa=init_qa,
        conf=conf,
        base_conf=base_conf,
        prompt_gen_data=prompt_gen_data,
        random_proj=random_proj,
        intrinsic_dim=intrinsic_dim,
        n_prompt_tokens=n_prompt_tokens,
        HF_cache_dir=HF_cache_dir,
        args=args,
    )

    # === 初始化 X / batch_caller / logger（务必在 eval_batch 之前）===
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    batch_caller = BatchLLMCaller(model_forward_api, max_workers=1)
    logger = TrainerLogger(log_dir="runs/instructzero_exp1", use_tensorboard=True)

    # 初始化 BO 追踪
    trace_path = _init_trace(task)
    
    # ================= [修改开始] =================
    # 初次评测
    X_return = batch_caller.eval_batch([x for x in X])

    # === 1. 解析返回值 (Y, S, Text) ===
    # 我们这里手动解包，不再只依赖 _unpack_eval_return，因为它原来只丢回两个值
    # 我们需要同时捕获：分数(y), 详细得分(s), 和 指令文本(text)
    Y_list = []
    S_list = []
    Instr_list = [] # <--- 新增：存指令文本

    for ret in X_return:
        y, s, text = 0.0, [], "" # 默认值
        
        # 情况A: 是 tuple/list (我们期望的情况，前提是你改了 LMForwardAPI)
        if isinstance(ret, (list, tuple)):
            if len(ret) >= 3:
                y, s, text = ret[0], ret[1], ret[2]
            elif len(ret) == 2:
                # 兼容旧接口，防止崩代码，但 text 依然会是空
                y, s = ret
            else:
                if len(ret) > 0: y = ret[0]
        
        # 情况B: 是 dict (防御性编程)
        elif isinstance(ret, dict):
            y = ret.get("score", ret.get("dev_perf", 0.0))
            s = ret.get("scores", [])
            text = ret.get("instruction", "")
            
        Y_list.append(y)
        S_list.append(s)
        Instr_list.append(text)

    # === 2. NaN/Inf 清洗 + 形状统一 ===
    # Y 统一成标量
    Y_list = [float(y) if (y is not None and math.isfinite(float(y))) else 0.0 for y in Y_list]

    # Y_scores 统一成等长向量（用 0 padding）
    S_list = [_np.nan_to_num(_np.asarray(s, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1) for s in S_list]
    max_len = max((arr.size for arr in S_list), default=0)
    S_list = [_np.pad(arr, (0, max_len - arr.size), mode='constant') for arr in S_list]

    # to tensors
    device = tkwargs["device"] if "device" in tkwargs else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs["device"] = device
    
    # 确保 X 也是 float64 (double)
    X = X.to(dtype=tkwargs['dtype'], device=device) 
    Y = torch.tensor(_np.asarray(Y_list), dtype=tkwargs['dtype'], device=device).unsqueeze(-1)
    Y_scores = torch.tensor(_np.stack(S_list, axis=0), dtype=tkwargs['dtype'], device=device)

    # === 3. 掩码过滤 (Masking) - 关键步骤 ===
    # 如果 Y 中有 NaN，我们必须把 Instr_list 对应的项也删掉，保持对齐
    mask = torch.isfinite(Y.squeeze(-1)) & torch.isfinite(Y_scores).all(dim=1)
    
    if not mask.all():
        bad = (~mask).nonzero(as_tuple=False).squeeze(-1).tolist()
        print(f"[EVAL WARNING] dropped {len(bad)} NaN/inf samples: {bad[:10]}")
        
        # 过滤 Tensor
        X = X[mask]
        Y = Y[mask]
        Y_scores = Y_scores[mask]
        
        # 过滤 List (指令文本)
        # 将 mask 转为 numpy bool 数组以便于列表过滤
        mask_np = mask.cpu().numpy().astype(bool)
        Instr_list = [inst for inst, m in zip(Instr_list, mask_np) if m]

    print(f"Best initial point: {Y.max().item():.3f}")

    # safe normalization of Y
    X_train = X
    Y_mean = Y.mean(dim=-2)
    Y_std = Y.std(dim=-2)
    if Y.shape[-2] <= 1 or (Y_std < 1e-3).all():
        y_train = Y - Y_mean
    else:
        y_train = (Y - Y_mean) / (Y_std + 1e-9)

    # === 4. [PATCH] 协变量历史 (修复版) ===
    # 现在我们有 Instr_list 了，可以正确计算协变量
    Y_raw_np = Y.squeeze(-1).detach().cpu().numpy()
    S_raw_np = Y_scores.detach().cpu().numpy()
    F_hist = []

    # 注意：Y_raw_np 已经被 mask 过滤过了，Instr_list 也被 mask 过滤过了
    # 所以它们的长度是严格一致的，可以直接 zip 或者通过 index 访问
    for j in range(Y_raw_np.shape[0]):
        # 获取真实的指令文本！
        # 如果前面没过滤好，这里就会 index out of range，所以上面的 mask 步骤很重要
        curr_instr = Instr_list[j] if j < len(Instr_list) else ""

        # 使用新的 8维 协变量系统
        # 注意：这里调用的是 compute_8d_covariates
        covariates_8d = semantic_covariate_system.compute_8d_covariates(
            instruction=curr_instr, 
            performance=Y_raw_np[j],
            scores=S_raw_np[j],
            step=0,
            total_steps=N_ITERATIONS,
            task_name=task
        )

        f8_new = covariates_8d.cpu().numpy()
        F_hist.append(f8_new)
    # 维护"当前最优指令文本/得分向量"
    best_prompt_text = ""
    best_S_row = None
    try:
        argmax0 = int(_np.argmax(Y_raw_np))
        best_S_row = S_raw_np[argmax0].copy()
        # 同时更新最优文本
        if argmax0 < len(Instr_list):
            best_prompt_text = Instr_list[argmax0]
    except Exception:
        pass
    # ================= [修改结束] =================
    # 也维护“当前最优指令文本/得分向量”，便于后续计算 sim/delta
    best_prompt_text = ""
    best_S_row = None
    try:
        argmax0 = int(_np.argmax(Y_raw_np))
        best_S_row = S_raw_np[argmax0].copy()
    except Exception:
        pass

    # === (A) 首次建 GP 前：按任务 profile 选择指令核，并预处理 S=Y_scores ===
    task_name = conf.get('evaluation', {}).get('task', task)
    profile = TASK_PROFILES.get(task_name, DEFAULT_PROFILE)

    S = Y_scores.clone()  # [N, m]
    # 列标准化（样本难度）
    col_mean = S.mean(dim=0, keepdim=True)
    col_std = S.std(dim=0, keepdim=True).clamp_min(1e-3)
    S = (S - col_mean) / col_std

    instr_kernel_type = profile["kernel"]
    if instr_kernel_type == 'cosine':
        # 行 L2 归一（仅 cosine 用）
        row_norm = S.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        S = S / row_norm

    # 训练张量
    latent_train = X_train.float().to(device)
    instruction_train = S.to(device)
    y_train = y_train.to(device)

    # initial GP construction with fallback
    try:
        gp_model, gp_mll = construct_custom_gp(
            X_train, y_train, latent_train, instruction_train, device,
            instr_kernel_type=instr_kernel_type
        )

        # 规则任务：不允许协变量主导
        allow_mix_opt = True
        _install_covariates_into_kernel(gp_model, F_hist, y_train, step_idx=0, allow_mix_opt=allow_mix_opt)

        # 规则任务把 alpha_cov 压到 0
        #if task_name in RULE_TASKS:
         #   try:
          #      kern = gp_model.covar_module.base_kernel
           #     kern.alpha_cov = 0.0
            #except Exception:
            #pass

        gp_model.train()
        with torch.no_grad():
            try:
                K_op = gp_model.covar_module(X_train[:3].double())
  # 这里返回的是 LinearOperator 或 Tensor
                # 尝试转稠密，只用于诊断输出，不影响训练
                if hasattr(K_op, "to_dense"):
                    K_dense = K_op.to_dense()
                elif torch.is_tensor(K_op):
                    K_dense = K_op
                else:
                    K_dense = None

                if K_dense is not None:
                    print("Custom kernel sample K shape:", K_dense.shape,
                          "has NaN:", torch.isnan(K_dense).any().item())
                else:
                    # 无法稠密化时，至少打印类型，避免抛错打断训练
                    print("Custom kernel operator:", type(K_op), "(skip densify)")
            except Exception as e:
                print("Kernel diagnostic skipped:", e)
        with gpytorch.settings.cholesky_jitter(1e-3):
            fit_gpytorch_mll(gp_mll)
    except Exception as e:
        print("[WARNING] initial custom kernel GP fit failed, falling back to baseline. Error:", e)
        gp_model, gp_mll = build_and_fit_simple_gp(X_train, y_train, device)

    # 收集已见过的指令（初次评测期间生成的）
    seen_instructions = set(model_forward_api.return_prompts_set().keys())

    try:
        for i in tqdm(range(N_ITERATIONS), desc="Bayes iterations"):
            logger.logger.info(f"[Iteration {i}] X_train {X_train.shape}, y_train {y_train.shape}")

            # 1. Fit GP
            with profile_block("GP fit", logger=logger.logger):
                # 每轮更新后的 latent/instruction/y
                latent_train = X_train.to(device)

                # (C) 每轮都用最新 Y_scores 预处理 S，并沿用 instr_kernel_type
                S = Y_scores.clone()
                col_mean = S.mean(dim=0, keepdim=True)
                col_std = S.std(dim=0, keepdim=True).clamp_min(1e-3)
                S = (S - col_mean) / col_std
                if instr_kernel_type == 'cosine':
                    row_norm = S.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
                    S = S / row_norm
                instruction_train = S.to(device)

                y_train = y_train.to(device)

                # 先用 custom kernel 拟合，失败 fallback baseline
                try:
                    gp_model, gp_mll = construct_custom_gp(
                        X_train, y_train, latent_train, instruction_train, device,
                        instr_kernel_type=instr_kernel_type
                    )

                    allow_mix_opt = task_name not in RULE_TASKS
                    _install_covariates_into_kernel(gp_model, F_hist, y_train, step_idx=i+1, allow_mix_opt=allow_mix_opt)

                    if task_name in RULE_TASKS:
                        try:
                            kern = gp_model.covar_module.base_kernel
                            kern.alpha_cov = 0.0
                        except Exception:
                            pass

                    gp_model.train()
                    with torch.no_grad():
                        try:
                            K_op = gp_model.covar_module(X_train[:3].double())
 # 这里返回的是 LinearOperator 或 Tensor
                            # 尝试转稠密，只用于诊断输出，不影响训练
                            if hasattr(K_op, "to_dense"):
                                K_dense = K_op.to_dense()
                            elif torch.is_tensor(K_op):
                                K_dense = K_op
                            else:
                                K_dense = None

                            if K_dense is not None:
                                print("Custom kernel sample K shape:", K_dense.shape,
                                      "has NaN:", torch.isnan(K_dense).any().item())
                            else:
                                # 无法稠密化时，至少打印类型，避免抛错打断训练
                                print("Custom kernel operator:", type(K_op), "(skip densify)")
                        except Exception as e:
                            print("Kernel diagnostic skipped:", e)
                    with gpytorch.settings.cholesky_jitter(1e-3):
                        fit_gpytorch_mll(gp_mll)
                except Exception as e:
                    print(f"[WARNING][Iteration {i}] custom kernel GP fit failed, falling back to baseline. Error: {e}")
                    gp_model, gp_mll = build_and_fit_simple_gp(X_train, y_train, device)

            current_best = y_train.max().item()
            gp_model.eval()

            # 计算当前的 marginal log likelihood
            with torch.no_grad():
                output = gp_model(X_train)
                try:
                    mll_value = gp_mll(output, y_train).sum().item()
                except Exception:
                    mll_value = float("nan")
            logger.log_iteration(i, best_value=current_best, gp_loss=mll_value)

# 2. Acquisition optimization（Log-EI→UCB兜底）
            with profile_block("Acquisition", logger=logger.logger):
                dim = X_train.shape[-1]
                # 确保 bounds 是 float64
                bounds = torch.stack([
                    torch.full((dim,), -1.0, device=device, dtype=torch.float64),
                    torch.full((dim,), 1.0, device=device, dtype=torch.float64)
                ])

                beta_scalar = 2.0 * (0.9 ** i)
                beta_tensor = torch.tensor([beta_scalar], dtype=tkwargs['dtype'], device=device) 
                UCB = UpperConfidenceBound(gp_model, beta=beta_tensor)
                
                # 初始化 acq_value 避免未定义
                acq_value = None

                try:
                    X_candidate, acq_value = optimize_acqf(
                        acq_function=UCB,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        options={"batch_limit": 5, "maxiter": 200},
                    )
                    # 【重要】确保类型匹配
                    X_candidate = X_candidate.to(dtype=torch.float64)
                    
                except Exception as e:
                    logger.logger.warning(f"UCB optimization failed: {e}")
                    # fallback 到随机
                    X_candidate = torch.rand(1, intrinsic_dim, device=device, dtype=torch.float64) * 2 - 1
                    acq_value = None

            # 【修复点】在此处定义 acq_val_num，供后面日志使用
            if acq_value is not None:
                if torch.is_tensor(acq_value):
                    acq_val_num = acq_value.item()
                else:
                    acq_val_num = float(acq_value)
            else:
                acq_val_num = 0.0

            # 3. Evaluate candidate (parallel)
            with profile_block("LLM eval candidate", logger=logger.logger):
                # ... (后续代码保持不变)
                candidate_tensor = X_candidate.detach().to(device)
                # 记录候选点后验
                with torch.no_grad():
                    post = gp_model.posterior(candidate_tensor)
                    mu = float(post.mean.squeeze().item())
                    sigma = float(post.variance.sqrt().squeeze().item())
                candidate_return = batch_caller.eval_batch([candidate_tensor.squeeze(0)])
            _cand = candidate_return[0] if isinstance(candidate_return, (list, tuple)) else candidate_return
            new_dev_perf, new_instruction_score = _unpack_eval_return(_cand)

            # —— 指令去重：若与已见相同，跳过追加 —— #
            instr_text = getattr(model_forward_api, "last_instruction_text", None)
            was_fallback = getattr(model_forward_api, "was_fallback", False)

            def _is_bad_instruction_outer(s: str, task_name: str) -> bool:
                if not s:
                    return True
                s = s.strip()
                if len(s) < 4 or len(s) > 220:
                    return True
                alpha_ratio = sum(ch.isalpha() for ch in s) / max(1, len(s))
                if alpha_ratio < 0.5:
                    return True
                sl = s.lower()
                # 避免把 few-shot DEMO 当作指令（对 antonyms 放宽你原有例外逻辑）
                if (task_name != "antonyms") and ("input:" in sl or "output:" in sl):
                    return True
                for bad in ("Too many requests", "As an AI", "I am sorry", "clarify your question"):
                    if bad.lower() in sl:
                        return True
                return False

            if was_fallback or _is_bad_instruction_outer(instr_text, task_name):
                logger.logger.info("Invalid/fallback instruction; skip appending to training set.")
                _append_trace(
                    trace_path, i, "bo-skip", new_dev_perf, new_dev_perf,
                    instr_text, None, _get_kernel_alphas(gp_model),
                    acq=acq_val_num, mu=mu, sigma=sigma
                )
                continue

            if instr_text in seen_instructions:
                logger.logger.info("Duplicate instruction detected; skip appending to training set.")
                _append_trace(
                    trace_path, i, "bo-skip", new_dev_perf, new_dev_perf,
                    instr_text, None, _get_kernel_alphas(gp_model),
                    acq=acq_val_num, mu=mu, sigma=sigma
                )
                continue

            seen_instructions.add(instr_text)

            # 归一化数值类型
            try:
                new_dev_perf = float(new_dev_perf)
                if not math.isfinite(new_dev_perf):
                    new_dev_perf = 0.0
            except Exception:
                new_dev_perf = 0.0

            ys = _np.nan_to_num(_np.asarray(new_instruction_score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)

            # 对齐列数
            curr_w = Y_scores.shape[1]
            if ys.size < curr_w:
                ys = _np.pad(ys, (0, curr_w - ys.size), mode='constant')
            elif ys.size > curr_w:
                # 扩宽已有矩阵到新的宽度
                pad_cols = ys.size - curr_w
                Y_scores = F.pad(Y_scores, (0, pad_cols))

            Y_next_point = torch.tensor([new_dev_perf], dtype=tkwargs['dtype'], device=device).unsqueeze(-1)
            Y_scores_next_point = torch.tensor(ys, dtype=tkwargs['dtype'], device=device).unsqueeze(0)
            X_next_point = candidate_tensor.to(dtype=tkwargs['dtype'], device=device)

      

            # 在BO循环中找到原来的协变量计算代码，替换为：

            # 使用新的 8维 语义协变量系统
            covariates_8d = semantic_covariate_system.compute_8d_covariates(
                instruction=instr_text,
                performance=new_dev_perf,
                scores=ys,
                step=i,
                total_steps=N_ITERATIONS,
                task_name=task
            )

            covariates_8d = covariates_8d.to(device)
            f8_new = covariates_8d.cpu().numpy()
            F_hist.append(f8_new)
            # 若成为新 best，则更新“当前最优指令文本/得分向量”
            try:
                if new_dev_perf > float(torch.max(Y).item()):
                    best_prompt_text = instr_text or best_prompt_text
                    best_S_row = ys.copy()
            except Exception:
                pass

            # ====== 追加到训练集 ======
            X = torch.cat([X, X_next_point], dim=0)
            Y = torch.cat([Y, Y_next_point], dim=0)
            Y_scores = torch.cat([Y_scores, Y_scores_next_point], dim=0)

            # 5. Rebuild GP with updated data
            X_train = X.clone()
            Y_mean = Y.mean(dim=-2)
            Y_std = Y.std(dim=-2)
            if Y.shape[-2] <= 1 or (Y_std < 1e-3).all():
                y_train = Y - Y_mean
            else:
                y_train = (Y - Y_mean) / (Y_std + 1e-9)

            logger.logger.info(f"Best value so far: {torch.max(Y).item():.5f}")
    finally:
        logger.close()

    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())

    # Evaluate on test data
    print('Evaluating on test data...')
    test_conf = get_test_conf(task, test_data)

    test_res = ape.evaluate_prompts(
        prompts=prompts,
        eval_template=eval_template,
        eval_data=test_data,
        few_shot_data=prompt_gen_data,
        demos_template=demos_template,
        conf=test_conf,
        base_conf=base_conf
    )
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    return test_score


if __name__ == '__main__':
    args = parse_args()
    print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")
    print(set_all_seed(args.seed))
    test_score = run(args=args)
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')
