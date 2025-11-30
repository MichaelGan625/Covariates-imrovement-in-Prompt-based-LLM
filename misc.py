import torch
import random
import numpy as np
from evaluation_instruction_induction.exec_accuracy import exec_accuracy_evaluator
import os

TASKS = [
    # === 性能严重下降组 (优先修复) ===
    'word_sorting',
    'word_unscrambling',
    'orthography_starts_with',
    'auto_categorization',
    'odd_one_out',

    # === 低分/无提升组 (次要修复) ===
    'auto_debugging',
    'synonyms',
    'second_word_letter'
]

SMOKE_TEST = os.environ.get("SMOKE_TEST")
## bayesian opt
tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,  # 统一用 float32
}

N_INIT = 25
N_ITERATIONS = 5 if not SMOKE_TEST else 1
BATCH_SIZE = 25 if not SMOKE_TEST else 1


def get_test_conf(task, test_data):
    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': 'meta-llama/llama-3-70b-instruct',  # 修改这里
                    'api_base': 'https://openrouter.ai/api/v1',
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'num_samples': min(100, len(test_data[0])),
            'task': task,
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                    'model': 'meta-llama/llama-3-70b-instruct',  # 修改这里
                    'api_base': 'https://openrouter.ai/api/v1',   
                }
            }
        }
    }
    return test_conf
def get_conf(task, eval_data):
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 10,
            'num_prompts_per_subsample': 20,
            'model': {
                'name': 'GPT_forward',  # <--- 必须添加这一行
                'gpt_config': {
                    'model': 'meta-llama/llama-3-70b-instruct',
                    'api_base': 'https://openrouter.ai/api/v1',
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'name': 'GPT_forward',  # <--- 必须添加这一行
                'gpt_config': {
                    'model': 'meta-llama/llama-3-70b-instruct',
                    'api_base': 'https://openrouter.ai/api/v1',
                }
            }
        }
    }
    return conf

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return f"Set all the seeds to {seed} successfully!"


# ==== misc.py 追加：四协变量特征 ====
import re
import numpy as np
from typing import List, Optional

_REFUSAL_KEYS = ["i'm sorry", "cannot", "不能", "无法", "作为", "作为一个"]
_CONSTRAINT_KEYS = ["only", "exactly", "do not", "without", "只", "不得", "必须", "仅输出", "只输出", "一词",
                    "single word"]


def _bow_vec(s: str):
    v = {}
    for w in re.findall(r"\w+", (s or "").lower()):
        v[w] = v.get(w, 0) + 1
    return v


def cosine_bow(a: str, b: str) -> float:
    va, vb = _bow_vec(a), _bow_vec(b)
    inter = set(va) & set(vb)
    dot = sum(va[w] * vb[w] for w in inter)
    na = sum(x * x for x in va.values()) ** 0.5
    nb = sum(x * x for x in vb.values()) ** 0.5
    return 0.0 if na == 0 or nb == 0 else float(dot / (na * nb))


def estimate_instruction_perplexity(prompt_text: str, scorer=None) -> float:
    """
    估计指令困惑度：
    - 若传入 scorer 且有 score_texts(texts)->neg_loglik/token_count 接口，则用之；
    - 否则用“长度代理”（log(len+1)）兜底，避免额外依赖。
    """
    if scorer is not None:
        try:
            # 期望：返回平均负对数似然（或 ppl），你可在本地 LLaMA/T5 上实现
            return float(scorer(prompt_text))
        except Exception:
            pass
    # 兜底：长度代理（越短越“易读”，给个单调替代）
    return float(np.log(len(prompt_text.strip()) + 1.0))


def predict_compliance_from_prompt(task: str, prompt_text: str) -> float:
    """
    在没有模型输出时，基于提示中的约束强度粗略预测“合规率”。
    强约束词越多、拒绝词越少、越简短 => 预测合规率越高。
    """
    p = (prompt_text or "").lower()
    cons = sum(k in p for k in _CONSTRAINT_KEYS)
    refus = sum(k in p for k in _REFUSAL_KEYS)
    L = max(1, len(prompt_text.strip()))
    score = cons * 1.0 - refus * 1.5 - 0.002 * L  # 线性打分
    # 映射到 (0,1)
    return float(1.0 / (1.0 + np.exp(-score)))


def compute_compliance_rate_from_preds(task: str, preds: List[str]) -> float:
    """
    用真实模型预测输出计算“合规率”：这里给通用规则：
    - 单词任务（antonyms/synonyms/taxonomy_animal/orthography_starts_with 等）：必须是单个词、无引号、无明显标点
    - 其他：只检查“不要解释性前缀”和“无拒绝词”
    若你有更细的任务正则，可在这里精炼。
    """
    if not preds:
        return 0.0
    single_word_like = {"antonyms", "synonyms", "taxonomy_animal", "orthography_starts_with"}
    punc = set(".,;:!?，。；：！？'\"`")
    cnt = 0
    for t in preds:
        s = (t or "").strip()
        s_low = s.lower()
        if any(k in s_low for k in _REFUSAL_KEYS):
            continue
        if task in single_word_like:
            tokens = s.split()
            if len(tokens) != 1:
                continue
            if any(c in punc for c in s):
                continue
        # 可以继续叠加你任务特有规则
        cnt += 1
    return float(cnt / len(preds))


def compute_four_covariates(
        prompt_text: str,
        task: str,
        preds: Optional[List[str]],
        answers: Optional[List[str]],
        S: Optional[np.ndarray],
        best_prompt_text: Optional[str],
        best_S: Optional[np.ndarray],
        ppl_scorer=None,
) -> np.ndarray:
    """
    4 维特征： [ sim_to_best, delta_best, -ppl, compliance ]
    - sim_to_best：与历史当前最优指令的 BoW 余弦相似度（无最优则 0）
    - delta_best：mean(S - best_S)，无 S/best_S 则 0
    - -ppl    ：指令困惑度取负号（困惑度越低越好 => 特征越大）
    - compliance：有 preds 就用真实合规率，否则用 prompt 结构预测
    """
    sim_to_best = cosine_bow(prompt_text, best_prompt_text or "")
    if S is not None and best_S is not None and len(S) == len(best_S):
        delta_best = float(np.mean(S - best_S))
        delta_best = max(delta_best, 0.0)  # 只强调“提升边际”
    else:
        delta_best = 0.0
    ppl = estimate_instruction_perplexity(prompt_text, scorer=ppl_scorer)
    neg_ppl = -float(ppl)
    if preds is not None:
        compliance = compute_compliance_rate_from_preds(task, preds)
    else:
        compliance = predict_compliance_from_prompt(task, prompt_text)
    return np.array([sim_to_best, delta_best, neg_ppl, compliance], dtype=np.float32)
