import numpy as np

from automatic_prompt_engineer import data, llm, evaluate
from evaluation_instruction_induction import utility


import os, csv, datetime, pathlib
import numpy as np

import os, csv, datetime, pathlib, re, hashlib  # 确保有 re, hashlib

def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _sanitize_filename(s: str, maxlen: int = 40) -> str:
    """把字符串净化成跨平台安全的文件名片段"""
    s = (s or "").replace("\n", " ").strip()
    # 去掉 Windows 禁止字符  \ / : * ? " < > |  和控制字符
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', ' ', s)
    # 再把非安全字符变为下划线
    s = re.sub(r'[^A-Za-z0-9._\- ]+', '_', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    if len(s) > maxlen:
        s = s[:maxlen]
    if not s:
        s = "snippet"
    return s

def _maybe_dump_preds(task_name, prompt_text, metric_name, inputs, golds, preds, scores):
    """
    把预测详细结果写到 TSV，并在日志里打印前 N 条。
    即使写文件失败，也会把前 N 条打印到日志里。
    环境变量：
      IZ_DUMP_PREDS_DIR:  保存目录（默认 logs/preds）
      IZ_DUMP_PREDS_N:    控制打印前 N 条（默认 5）
    """
    dump_dir = os.getenv("IZ_DUMP_PREDS_DIR", "logs/preds")
    _ensure_dir(dump_dir)
    tag = f"{task_name}_{_now_tag()}"

    # 文件名里的 prompt 片段用净化后文本 + hash（避免过长/重复）
    prompt_snippet_raw = (prompt_text or "").replace("\n", " ")
    prompt_snippet = _sanitize_filename(prompt_snippet_raw, maxlen=40)
    prompt_hash = hashlib.md5(prompt_snippet_raw.encode("utf-8")).hexdigest()[:8]
    fname = f"{tag}__{metric_name}__{prompt_snippet}__{prompt_hash}.tsv"
    tsv_path = os.path.join(dump_dir, fname)

    # 无论是否写文件成功，先在日志里预览前 N 条
    try:
        n = int(os.getenv("IZ_DUMP_PREDS_N", "5"))
    except Exception:
        n = 5
    n = max(0, n)
    for i in range(min(n, len(preds))):
        try:
            sc = float(scores[i])
        except Exception:
            sc = 0.0
        print(f"[PRED {i}] gold={golds[i]!r} | pred={preds[i]!r} | score={sc:.4f}")

    # 再尝试写 TSV；如果失败，不影响主流程
    try:
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["idx", "task", "metric", "score", "input", "gold", "pred", "prompt"])
            for i, (inp, gold, pred, sc) in enumerate(zip(inputs, golds, preds, scores)):
                try:
                    scf = float(sc)
                except Exception:
                    scf = 0.0
                w.writerow([i, task_name, metric_name, f"{scf:.4f}", str(inp), str(gold), str(pred), (prompt_text or "")])
        print(f"[PRED-DUMP] wrote {len(preds)} rows to {tsv_path}")
    except Exception as e:
        print(f"[PRED-DUMP] skip write ({e}) -> still printed top-{n} above")


# -------- helpers --------
def _normalize_text(x):
    try:
        s = str(x).strip()
        # 去掉首尾引号
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        return s.lower()
    except Exception:
        return x

def _get_score_fn(task, metric):
    """
    metric 可以是字符串或可调用；字符串优先在内置别名与 registry 中解析。
    """
    if callable(metric):
        return metric

    if isinstance(metric, str):
        # 内置别名
        alias = {
            'f1': utility.get_multi_answer_f1,
            'es': utility.get_multi_answer_exact_set,
            'contains': utility.get_multi_answer_contains,
            'em': utility.get_multi_answer_em,
            # 你日志里出现的名字：
            'metric_exact_or_contains': None,  # 尝试由 registry 解析
        }
        if metric in alias and alias[metric] is not None:
            return alias[metric]

        # 走 registry（automatic_prompt_engineer.metrics_registry）
        try:
            fn = resolve_metric(metric)
            if fn is not None:
                return fn
        except Exception:
            pass

    # 兜底
    return utility.get_multi_answer_em


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template, task_name=None):
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)

def get_query_for_test(prompt, eval_template, input_, output_, task_name=None):
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo='')
    t = (task_name or '').lower()
    if t == "cause_and_effect":
        query += "\n\nReturn ONLY the final answer, with no explanation. Use a single phrase if applicable."
    elif "synonym" in t or t == "synonyms":
        query += "\n\nReturn ONLY the single best synonym as one word. Do NOT include quotes, punctuation, or extra text."
    return query


def exec_accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    queries = []
    answers = []
    inputs = []
    task = config.get('task', '')

    # 固定测试数据 - 在prompt循环外采样
    fixed_test_data = data.subsample_data(eval_data, config['num_samples'])

    for prompt in prompts:
        for d in zip(*fixed_test_data):  # 所有prompt用相同的数据
            input_, output_ = d
            demo_data = data.subsample_data(few_shot_data, config['num_few_shot'])
            query = get_query(prompt, eval_template, input_, output_, demo_data, demos_template, task_name=task)
            queries.append(query)
            answers.append(output_)
            inputs.append(input_)

    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)

    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
    print(f'Using metric "{metric}" for task "{task}"...')

    score_fn = _get_score_fn(task, metric)

    # 评分
    scores = []
    for i, (prediction, ans_) in enumerate(zip(model_outputs, answers)):
        pred_use = _normalize_text(prediction)
        try:
            s = score_fn(pred_use, ans_)
        except Exception:
            s = 0.0
        # 清洗 NaN/inf
        try:
            if s is None:
                s = 0.0
            s = float(s)
            if not np.isfinite(s):
                s = 0.0
        except Exception:
            s = 0.0
        scores.append(s)

    # Reshape the scores so that it is num_prompts x num_samples
    flat_scores = list(scores)  # 还没 reshape 前的扁平分数
    n = config['num_samples']
    metric_name = metric if isinstance(metric, str) else getattr(score_fn, "__name__", str(metric))

    for p_idx, prompt in enumerate(prompts):
        s, e = p_idx * n, (p_idx + 1) * n
        preds_chunk = [_normalize_text(x) for x in model_outputs[s:e]]
        golds_chunk = answers[s:e]
        inputs_chunk = inputs[s:e]
        scores_chunk = flat_scores[s:e]
        try:
            _maybe_dump_preds(task, prompt, metric_name, inputs_chunk, golds_chunk, preds_chunk, scores_chunk)
        except Exception as _e:
            print(f"[PRED-DUMP] skipped ({_e})")

    # Reshape the scores so that it is num_prompts x num_samples
    scores = np.array(flat_scores).reshape(len(prompts), n)

    res = ExecAccuracyEvaluationResult(prompts, scores)
    return res, scores


class exec_evaluator(object):
    def __init__(self, api_model, config):
        # instantiate the LLM here
        if api_model=='llama':
            self.model = llm.Llama_Forward(config)
        elif api_model=='flan-t5':
            self.model = llm.Flan_T5(config)

    def evaluate(self, prompts, eval_template, eval_data, demos_template, few_shot_data, config):
        queries = []
        answers = []
        inputs = []
        # ⚠️ 不再强制把同一个 prompt 复制 20 次；按真实传入的 prompts 来
        task = config.get('task', '')

        for prompt in prompts:
            subsampled_data = data.subsample_data(
                eval_data, config['num_samples'])
            for d in zip(*subsampled_data):
                input_, output_ = d
                demo_data = data.subsample_data(
                    few_shot_data, config['num_few_shot'])
                query = get_query(
                    prompt, eval_template, input_, output_, demo_data, demos_template, task_name=task)
                queries.append(query)
                answers.append(output_)
                inputs.append(input_)  # NEW

        model_outputs = self.model.generate_text(queries, 1)

        metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
        print(f'Using metric "{metric}" for task "{task}"...')

        score_fn = _get_score_fn(task, metric)

        scores = []
        for prediction, ans_ in zip(model_outputs, answers):
            pred_use = _normalize_text(prediction)
            try:
                score = score_fn(pred_use, ans_)
            except Exception:
                score = 0.0
            try:
                score = float(score)
                if not np.isfinite(score):
                    score = 0.0
            except Exception:
                score = 0.0
            scores.append(score)

        # Reshape the scores so that it is num_prompts x num_samples
        flat_scores = list(scores)
        n = config['num_samples']
        metric_name = metric if isinstance(metric, str) else getattr(score_fn, "__name__", str(metric))
        for p_idx, prompt in enumerate(prompts):
            s, e = p_idx * n, (p_idx + 1) * n
            preds_chunk = [_normalize_text(x) for x in model_outputs[s:e]]
            golds_chunk = answers[s:e]
            inputs_chunk = inputs[s:e]
            scores_chunk = flat_scores[s:e]
            try:
                _maybe_dump_preds(task, prompt, metric_name, inputs_chunk, golds_chunk, preds_chunk, scores_chunk)
            except Exception as _e:
                print(f"[PRED-DUMP] skipped ({_e})")

        scores = np.array(flat_scores).reshape(len(prompts), n)
        res = ExecAccuracyEvaluationResult(prompts, scores)
        return res

    def test(self, prompts, eval_template, eval_data, config):
        queries = []
        answers = []
        num_samples = config['evaluation']['num_samples']
        task = config['evaluation'].get('task', '')

        for prompt in prompts:
            subsampled_data = data.subsample_data(
                eval_data, num_samples)
            for d in zip(*subsampled_data):
                input_, output_ = d
                query = get_query_for_test(
                    prompt, eval_template, input_, output_, task_name=task)
                queries.append(query)
                answers.append(output_)

        model_outputs = self.model.generate_text(queries, 1)

        metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
        print(f'Using metric "{metric}" for task "{task}"...')

        score_fn = _get_score_fn(task, metric)

        scores = []
        for prediction, ans_ in zip(model_outputs, answers):
            pred_use = _normalize_text(prediction)
            try:
                score = score_fn(pred_use, ans_)
            except Exception:
                score = 0.0
            try:
                score = float(score)
                if not np.isfinite(score):
                    score = 0.0
            except Exception:
                score = 0.0
            scores.append(score)

        # Reshape the scores so that it is num_prompts x num_samples
        scores = np.array(scores).reshape(len(prompts), num_samples)
        res = ExecAccuracyEvaluationResult(prompts, scores)
        return res


class ExecAccuracyEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""

        def _safe(arr, fn):
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                return 0.0
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return float(fn(a))

        if method == 'mean':
            return [_safe(s, np.mean) for s in self.scores]
        elif method == 'median':
            return [_safe(s, np.median) for s in self.scores]
        elif method == 'std':
            return [_safe(s, np.std) for s in self.scores]
        elif method == 'max':
            return [_safe(s, np.max) for s in self.scores]
        elif method == 'min':
            return [_safe(s, np.min) for s in self.scores]
        elif method == 'iqm':
            vals = []
            for s in self.scores:
                a = np.asarray(s, dtype=float)
                if a.size == 0:
                    vals.append(0.0);
                    continue
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                q1, q3 = np.percentile(a, [25, 75])
                trimmed = a[(a >= q1) & (a <= q3)]
                vals.append(float(trimmed.mean()) if trimmed.size else float(a.mean()))
            return vals
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores