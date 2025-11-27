# utility.py  —— 更稳健的归一化 + 指标实现（含 exact_or_contains）
import re
import re
import unicodedata

# =====================
# 文本归一化（对模型输出 & 标准答案都用）
# =====================
_SEP_PATTERN = re.compile(r"[,\|;/]")
_UNITS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}

def _wordnum_to_int_token(tok: str):
    tok = tok.strip().lower()
    if tok in _UNITS: return _UNITS[tok]
    if tok in _TENS: return _TENS[tok]
    if "-" in tok:  # e.g., seventy-eight
        a,b = tok.split("-",1)
        if a in _TENS and b in _UNITS:
            return _TENS[a] + _UNITS[b]
    return None

def _normalize_nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("−","-").replace(",", " ")

def _find_numbers_mixed(s: str):
    """
    从字符串中按顺序提取所有数字：
      - 阿拉伯数字：-?\d+
      - 英文写法（zero..ninety-nine，含 hyphen）：支持可选负号
    返回 List[int]
    """
    s = _normalize_nfkc(str(s))
    nums = []
    # 先抓阿拉伯数字
    for m in re.finditer(r"[-+]?\d+", s):
        try: nums.append(int(m.group(0)))
        except: pass
    # 再抓英文写法（带可选负号），形如 "ninety" 或 "seventy-eight"
    for m in re.finditer(r"[-+]?\s*[A-Za-z]+(?:-[A-Za-z]+)?", s):
        tok = m.group(0).strip()
        sign = -1 if tok.startswith("-") else 1
        tok = tok.lstrip("+- ").strip()
        val = _wordnum_to_int_token(tok)
        if val is not None:
            nums.append(sign * val)
    return nums

def _extract_gold_ints(gold):
    """
    gold 可能是 str 或 List[str]；返回所有能解析出的整数候选
    """
    if isinstance(gold, (list, tuple)):
        cands = []
        for g in gold:
            cands.extend(_find_numbers_mixed(str(g)))
        return cands
    return _find_numbers_mixed(str(gold))

def _extract_first_int(s):
    """从字符串里提取首个“有符号整数”，支持 NFKC、括号负号、U+2212 等。"""
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", str(s))
    s = s.strip()
    s = s.replace(",", "")          # 去千分位
    s = s.replace("−", "-")         # U+2212 → ASCII -
    # 允许括号包裹、可选正负号、可选小数点（如 5.）
    m = re.search(r"[-+]?\s*\(?\s*\d+\s*\.?\s*\)?", s)
    if not m:
        return None
    tok = m.group(0)
    tok = tok.strip().strip("()").strip()
    if tok.endswith("."):
        tok = tok[:-1]
    try:
        return int(tok)
    except Exception:
        return None

def diff_numeric_em(pred, gold):
    """
    如果 pred 里有两个数（数字或英文单词），按 first - second 计算结果；
    否则尝试把 pred 当作“直接给出的单个结果”。
    只要和 gold 的任一候选整数相等，返回 1.0；否则 0.0
    """
    nums = _find_numbers_mixed(pred)
    if len(nums) >= 2:
        got = nums[0] - nums[1]
    elif len(nums) == 1:
        got = nums[0]
    else:
        return 0.0
    gold_ints = _extract_gold_ints(gold)
    return 1.0 if got in gold_ints else 0.0

def sum_numeric_em(pred, gold):
    nums = _find_numbers_mixed(pred)
    if len(nums) >= 2:
        got = nums[0] + nums[1]
    elif len(nums) == 1:
        got = nums[0]
    else:
        return 0.0
    gold_ints = _extract_gold_ints(gold)
    return 1.0 if got in gold_ints else 0.0

def _first_line(s: str) -> str:
    s = s.strip()
    # 去掉常见前缀
    s = re.sub(r'^\s*(final answer|answer|output)\s*[:：]\s*', '', s, flags=re.I)
    # 取第一行
    s = s.splitlines()[0] if "\n" in s else s
    return s.strip()

def _normalize_token(s: str) -> str:
    """
    归一化到“单词答案”场景：小写、去引号括号标点，只保留字母，取第一个词。
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.strip('"\''" \t`")
    s = _first_line(s)
    # 去代码围栏
    if s.startswith("```"):
        s = s.strip("`").strip()
    # 只保留 a-z，用空格替换其它，再取第一个词
    s = re.sub(r"[^a-z]+", " ", s).strip()
    # 有些模型会给多个候选，取第一个词
    s = s.split()[0] if s else ""
    return s

def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _normalize_gold(ans):
    """
    标准答案可能是单个字符串或列表；也可能带分隔符（/,|,;）。
    这里统一拆分 -> 归一化 -> 去空项。
    """
    out = []
    for a in _to_list(ans):
        if a is None:
            continue
        a = str(a)
        # 先按常见分隔符拆
        parts = _SEP_PATTERN.split(a) if _SEP_PATTERN.search(a) else [a]
        for p in parts:
            tok = _normalize_token(p)
            if tok:
                out.append(tok)
    # 去重
    out = list(dict.fromkeys(out))
    return out

# =====================
# 指标函数
# =====================
def get_multi_answer_em(pred, gold):
    p = _normalize_token(pred)
    golds = _normalize_gold(gold)
    if not p or not golds:
        return 0.0
    return 1.0 if p in golds else 0.0

def get_multi_answer_contains(pred, gold):
    """
    是否包含正确 token（按词边界；对长输出稍放宽，但依然以词为单位）。
    """
    p = _normalize_token(pred)  # 单词级任务我们只看第一个词
    golds = _normalize_gold(gold)
    if not p or not golds:
        return 0.0
    # 允许 gold 的任一候选命中
    return 1.0 if any(p == g for g in golds) else 0.0

def _bag_of_words_tokens(s: str):
    s = str(s or "").lower()
    s = _first_line(s)
    # 粗略分词：保留字母和空格
    s = re.sub(r"[^a-z]+", " ", s).strip()
    toks = [t for t in s.split() if t]
    return toks

def get_multi_answer_f1(pred, gold):
    """
    词袋 F1（对句子/短语任务较稳，对单词任务也能兜底）。
    """
    p_toks = _bag_of_words_tokens(pred)
    # gold 取所有候选里“最佳匹配”的那个（max F1）
    best = 0.0
    for g in _to_list(gold):
        g_toks = _bag_of_words_tokens(g)
        if not p_toks and not g_toks:
            f1 = 1.0
        elif not p_toks or not g_toks:
            f1 = 0.0
        else:
            p_set, g_set = set(p_toks), set(g_toks)
            tp = len(p_set & g_set)
            prec = tp / max(len(p_set), 1)
            rec  = tp / max(len(g_set), 1)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return float(best)

def get_multi_answer_exact_set(pred, gold):
    """
    对需要“集合完全一致”的任务（如按字母开头列举 token）。
    """
    p_set = set(_bag_of_words_tokens(pred))
    best = 0.0
    for g in _to_list(gold):
        g_set = set(_bag_of_words_tokens(g))
        best = max(best, 1.0 if p_set == g_set else 0.0)
    return best

def get_multi_answer_exact_or_contains(pred, gold):
    """
    专为‘同义词/单词级’：先 exact（EM），不行再 contains。
    """
    em = get_multi_answer_em(pred, gold)
    if em == 1.0:
        return 1.0
    return get_multi_answer_contains(pred, gold)

# =====================
# 任务到指标的映射
# =====================
TASK_TO_METRIC = {
    'common_concept': 'f1',
    'informal_to_formal': 'f1',
    'orthography_starts_with': 'es',
    'taxonomy_animal': 'es',
    # 关键：同义词 -> 更稳的 exact_or_contains
    'synonyms': 'exact_or_contains',
    'cause_and_effect': 'f1',
    'auto_categorization': 'f1',
    'letters_list': 'em',
    'second_word_letter': 'em',
    'periodic_elements': 'em',
    'diff': 'diff_numeric_em',
    'sum': 'sum_numeric_em'
}

default_metric = 'em'