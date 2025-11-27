# semantic_covariates.py
import torch
import numpy as np
import re
from typing import List, Dict, Deque
from collections import deque, Counter
import math


class SemanticCovariateSystem:
    """
    12维语义协变量系统
    每个维度都有明确的语义含义，避免数学扩展的冗余
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history

        # 历史记录
        self.performance_history = deque(maxlen=max_history)
        self.instruction_history = deque(maxlen=max_history)
        self.score_history = deque(maxlen=max_history)

        # 任务关键词库（可根据任务动态扩展）
        self.task_keywords = {
            'synonyms': ['synonym', 'similar', 'same meaning', 'equivalent', 'word'],
            'antonyms': ['antonym', 'opposite', 'contrary', 'reverse', 'word'],
            'cause_and_effect': ['cause', 'effect', 'reason', 'result', 'because', 'leads to'],
            'common_concept': ['common', 'concept', 'category', 'group', 'shared', 'theme'],
            'default': ['find', 'return', 'identify', 'list', 'generate']
        }

    def compute_12d_covariates(self, instruction: str, performance: float,
                               scores: np.ndarray, step: int, total_steps: int,
                               task_name: str = None) -> torch.Tensor:
        """
        计算12维语义协变量
        """
        covariates = torch.zeros(12, dtype=torch.float32)

        # 更新历史记录
        self._update_history(instruction, performance, scores)

        # 1. 性能相关维度 (3维)
        covariates[0] = self._absolute_performance(performance)
        covariates[1] = self._relative_improvement(performance)
        covariates[2] = self._performance_consistency(scores)

        # 2. 指令质量维度 (3维)
        covariates[3] = self._instruction_clarity(instruction)
        covariates[4] = self._constraint_appropriateness(instruction)
        covariates[5] = self._task_alignment(instruction, task_name)

        # 3. 搜索状态维度 (3维)
        covariates[6] = self._search_stage(step, total_steps)
        covariates[7] = self._exploration_need(step, total_steps)
        covariates[8] = self._region_coverage()

        # 4. 风险预警维度 (3维)
        covariates[9] = self._convergence_risk()
        covariates[10] = self._instruction_novelty(instruction)
        covariates[11] = self._output_reliability(scores)

        return covariates

    def _update_history(self, instruction: str, performance: float, scores: np.ndarray):
        """更新历史记录"""
        self.instruction_history.append(instruction)
        self.performance_history.append(performance)
        self.score_history.append(scores)

    # === 性能相关维度 ===

    def _absolute_performance(self, performance: float) -> torch.Tensor:
        """绝对性能水平"""
        # 直接归一化到0-1范围，返回 tensor
        perf_value = min(1.0, max(0.0, performance))
        return torch.tensor(perf_value, dtype=torch.float32)

    def _relative_improvement(self, current_performance: float) -> torch.Tensor:
        """相对改进幅度"""
        if len(self.performance_history) < 3:
            return torch.tensor(0.5, dtype=torch.float32)  # 中性值

        # 计算近期性能基准
        recent_performances = list(self.performance_history)[-5:]
        baseline = np.mean(recent_performances[:-1])  # 排除当前点

        if baseline < 1e-6:
            return torch.tensor(0.5, dtype=torch.float32)

        # 标准化改进
        improvement = (current_performance - baseline) / baseline
        # 使用tanh映射到[-0.5, 0.5]，然后调整到[0, 1]
        normalized = 0.5 + 0.5 * np.tanh(improvement * 2.0)

        return torch.tensor(float(normalized), dtype=torch.float32)

    def _performance_consistency(self, scores: np.ndarray) -> torch.Tensor:
        """性能一致性"""
        if len(scores) < 2:
            return torch.tensor(0.8, dtype=torch.float32)  # 单样本时假设稳定

        std = np.std(scores)
        # 低标准差表示高一致性
        consistency = 1.0 - min(1.0, std * 3.0)  # 假设std正常范围在0-0.33
        return torch.tensor(max(0.1, consistency), dtype=torch.float32)

    # 对其他所有返回标量值的方法也做同样的修改：
    def _instruction_clarity(self, instruction: str) -> torch.Tensor:
        """指令清晰度"""
        score = 0.0

        # 长度适当性 (40%)
        length = len(instruction)
        if 20 <= length <= 100:
            score += 0.4
        elif 10 <= length < 20 or 100 < length <= 150:
            score += 0.2
        else:
            score += 0.1

        # 结构清晰度 (30%)
        sentences = re.split(r'[.!?]+', instruction)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        if 1 <= len(valid_sentences) <= 3:
            score += 0.3
        elif len(valid_sentences) == 0:
            score += 0.1
        else:
            score += 0.2

        # 标点使用 (30%)
        if instruction.strip().endswith(('.', '?')):
            score += 0.3
        elif instruction.strip().endswith(','):
            score += 0.1

        return torch.tensor(min(1.0, score), dtype=torch.float32)

    def _constraint_appropriateness(self, instruction: str) -> torch.Tensor:
        """约束适当性"""
        strong_constraints = ['must', 'only', 'never', 'always', 'exactly']
        moderate_constraints = ['should', 'avoid', 'prefer', 'try to']
        weak_constraints = ['can', 'may', 'consider', 'optionally']

        strong_count = sum(1 for c in strong_constraints if c in instruction.lower())
        moderate_count = sum(1 for c in moderate_constraints if c in instruction.lower())
        weak_count = sum(1 for c in weak_constraints if c in instruction.lower())

        # 加权计数
        total_constraint = strong_count * 0.5 + moderate_count * 0.3 + weak_count * 0.2
        appropriateness = min(1.0, total_constraint * 0.7)  # 适当缩放

        return torch.tensor(appropriateness, dtype=torch.float32)

    def _task_alignment(self, instruction: str, task_name: str) -> float:
        """任务对齐度"""
        if not task_name:
            return 0.5

        # 获取任务关键词
        keywords = self.task_keywords.get(task_name, self.task_keywords['default'])

        # 计算关键词匹配度
        instruction_lower = instruction.lower()
        matches = sum(1 for keyword in keywords if keyword in instruction_lower)
        match_ratio = matches / len(keywords)

        # 考虑指令的明确性
        action_verbs = ['return', 'find', 'identify', 'list', 'generate', 'create']
        has_action = any(verb in instruction_lower for verb in action_verbs)

        alignment = match_ratio * 0.7 + (0.3 if has_action else 0.1)
        return min(1.0, alignment)

    # === 搜索状态维度 ===

    def _search_stage(self, step: int, total_steps: int) -> float:
        """搜索阶段"""
        progress = step / total_steps

        # 映射到三个阶段
        if progress < 0.3:
            return 0.3  # 早期
        elif progress < 0.7:
            return 0.6  # 中期
        else:
            return 0.9  # 后期

    def _exploration_need(self, step: int, total_steps: int) -> float:
        """探索需求"""
        base_need = 1.0 - (step / total_steps)  # 随进度减少

        # 如果性能停滞，增加探索需求
        if len(self.performance_history) >= 5:
            recent_perf = list(self.performance_history)[-5:]
            if max(recent_perf) - min(recent_perf) < 0.05:  # 停滞阈值
                base_need = min(1.0, base_need + 0.3)

        # 如果多样性低，增加探索需求
        if len(self.instruction_history) >= 3:
            diversity = self._calculate_diversity()
            if diversity < 0.3:
                base_need = min(1.0, base_need + 0.2)

        return base_need

    def _region_coverage(self) -> float:
        """区域覆盖度"""
        if len(self.instruction_history) < 3:
            return 0.5

        # 计算指令多样性作为覆盖度代理
        diversity = self._calculate_diversity()
        return diversity

    def _calculate_diversity(self) -> float:
        """计算指令多样性"""
        if len(self.instruction_history) < 2:
            return 1.0

        similarities = []
        instructions = list(self.instruction_history)

        for i in range(len(instructions)):
            for j in range(i + 1, len(instructions)):
                sim = self._cosine_similarity(instructions[i], instructions[j])
                similarities.append(sim)

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        return max(0.1, diversity)

    # === 风险预警维度 ===

    def _convergence_risk(self) -> float:
        """收敛风险"""
        if len(self.performance_history) < 5:
            return 0.1  # 早期风险低

        recent_perf = list(self.performance_history)[-5:]

        # 性能停滞风险
        improvements = [recent_perf[i] - recent_perf[i - 1] for i in range(1, len(recent_perf))]
        stagnation = sum(1 for imp in improvements if abs(imp) < 0.01) / len(improvements)

        # 方差下降风险
        variance = np.var(recent_perf)
        variance_risk = 1.0 - min(1.0, variance * 10.0)

        # 组合风险
        total_risk = (stagnation + variance_risk) / 2.0
        return min(1.0, total_risk)

    def _instruction_novelty(self, instruction: str) -> float:
        """指令新颖性"""
        if len(self.instruction_history) < 2:
            return 1.0  # 第一个指令最高新颖性

        # 计算与最近指令的最大相似度
        recent_instructions = list(self.instruction_history)[-5:]  # 最近5个
        max_similarity = 0.0

        for prev_instr in recent_instructions:
            if prev_instr == instruction:
                continue
            similarity = self._cosine_similarity(instruction, prev_instr)
            max_similarity = max(max_similarity, similarity)

        novelty = 1.0 - max_similarity
        return max(0.1, novelty)

    def _output_reliability(self, scores: np.ndarray) -> float:
        """输出可靠性"""
        if len(scores) < 2:
            return 0.7  # 单样本时假设中等可靠性

        # 检查极端值比例
        extreme_low = sum(1 for s in scores if s < 0.1)
        extreme_high = sum(1 for s in scores if s > 0.9)
        extreme_ratio = (extreme_low + extreme_high) / len(scores)

        # 检查一致性模式
        std = np.std(scores)

        # 组合可靠性指标
        reliability = (1.0 - extreme_ratio) * 0.6 + (1.0 - min(1.0, std * 2.0)) * 0.4
        return max(0.1, reliability)

    def _cosine_similarity(self, a: str, b: str) -> float:
        """计算词袋余弦相似度"""

        def get_bow(text):
            words = re.findall(r'\w+', text.lower())
            return Counter(words)

        vec_a = get_bow(a)
        vec_b = get_bow(b)

        common_words = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[word] * vec_b[word] for word in common_words)

        norm_a = math.sqrt(sum(count ** 2 for count in vec_a.values()))
        norm_b = math.sqrt(sum(count ** 2 for count in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)