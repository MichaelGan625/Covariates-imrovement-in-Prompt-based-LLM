import os
import yaml


def update_config(config, base_config='configs/default.yaml'):
    # 构造默认配置文件路径（相对 current file）
    default_config_path = os.path.join(os.path.dirname(__file__), base_config)
    default_config_path = os.path.abspath(default_config_path)
    if not os.path.isfile(default_config_path):
        raise FileNotFoundError(f"Base config file not found: {default_config_path}")

    # 用 UTF-8 打开以避免编码错误
    with open(default_config_path, encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    # 递归更新字典
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, config)

# config.py
def simple_config(eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size):
    """Returns a config and splits the data into sensible chunks."""
    conf = update_config({}, 'configs/bandits.yaml')
    conf['generation']['model']['gpt_config']['model'] = prompt_gen_model
    if prompt_gen_mode == 'insert':
        conf['generation']['model']['name'] = 'GPT_insert'
        conf['generation']['model']['batch_size'] = 1
    elif prompt_gen_mode == 'forward':
        conf['generation']['model']['name'] = 'GPT_forward'
        conf['generation']['model']['batch_size'] = prompt_gen_batch_size
    conf['generation']['num_subsamples'] = num_prompts // 10
    conf['generation']['num_prompts_per_subsample'] = 10

    conf['evaluation']['base_eval_config']['model']['gpt_config']['model'] = eval_model
    conf['evaluation']['base_eval_config']['model']['batch_size'] = eval_batch_size
    conf['evaluation']['num_prompts_per_round'] = 0.334
    conf['evaluation']['rounds'] = eval_rounds
    conf['evaluation']['base_eval_config']['num_samples'] = 15
    
    return conf
