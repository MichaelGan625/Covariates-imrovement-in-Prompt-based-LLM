import subprocess
import os
import re
import csv
from datetime import datetime

# 任务列表（跟 misc.py 一致）
TASKS = [
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'word_in_context', 'auto_categorization', 'auto_debugging',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out'
]


# 固定参数（你原始命令模版）
MODEL_NAME = "vicuna"
HF_CACHE_DIR = r"D:\Py\LLM"
RANDOM_PROJ = "uniform"
INTRINSIC_DIM = "30"
N_PROMPT_TOKENS = "5"
SEED = "42"
API_MODEL = "chatgpt"

# ===== 仅保留自动学习与开关，不传手动权重 =====
USE_SCALARIZATION = True   # 启用协变量标量化（奖励+软约束）
AUTO_WEIGHTS      = True   # 让程序按任务在线学习权重
ANNEAL_DIVERSITY  = True   # 多样性权重退火

# 输出目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

summary = []

for task in TASKS:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{task}_{timestamp}.log")
    cmd = [
        "python", "run_instructzero.py",
        "--task", task,
        "--model_name", MODEL_NAME,
        "--api_model", API_MODEL,
        "--HF_cache_dir", HF_CACHE_DIR,
        "--random_proj", RANDOM_PROJ,
        "--intrinsic_dim", INTRINSIC_DIM,
        "--n_prompt_tokens", N_PROMPT_TOKENS,
        "--seed", SEED,
    ]


    print(f"\n=== Running task: {task} ===")
    print(" ".join(cmd))
    exit_code = None
    score = "N/A"
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            exit_code = proc.returncode
        # 解析 log 提取 score
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            m = re.search(r"Test score on ChatGPT:\s*([0-9]*\.?[0-9]+)", content)
            if m:
                score = m.group(1)
            else:
                m2 = re.search(r"best_value=([0-9]*\.?[0-9]+)", content)
                if m2:
                    score = m2.group(1)
        if exit_code == 0 and score != "N/A":
            status = "Success"
        elif exit_code == 0:
            status = "Partial"
        else:
            status = "Fail"
        print(f"Task={task} exit_code={exit_code} status={status} score={score}")
    except Exception as e:
        exit_code = -1
        status = "Fail"
        print(f"[ERROR] Task={task} exception: {e}")
    summary.append({
        "task": task,
        "status": status,
        "score": score,
        "exit_code": exit_code,
        "log": log_file,
    })

# 写 summary csv
summary_path = os.path.join(log_dir, "summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=["task", "status", "score", "exit_code", "log"])
    writer.writeheader()
    for row in summary:
        writer.writerow(row)

# 打印表格
print("\n===== Summary =====")
print(f"{'task':30} {'status':10} {'score':8} {'exit_code':9} {'log'}")
for row in summary:
    print(f"{row['task']:30} {row['status']:10} {row['score']:8} {str(row['exit_code']):9} {row['log']}")

print(f"\nSaved summary to {summary_path}")
