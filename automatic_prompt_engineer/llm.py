"""Contains classes for querying large language models."""
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai
import torch
import asyncio
from typing import Any
from openai import OpenAI
import os
import asyncio
try:
    from openai import OpenAI  # new SDK
    HAS_NEW_OPENAI = True
except ImportError:
    import openai  # fallback old style
    HAS_NEW_OPENAI = False

# 在 llm.py 中更新成本字典
gpt_costs_per_thousand = {
    'davinci': 0.0200,
    'curie': 0.0020, 
    'babbage': 0.0005,
    'ada': 0.0004,
    'meta-llama/llama-3-70b-instruct': 0.00059,  # Llama 3 70B成本
    'microsoft/wizardlm-2-8x22b': 0.00065,       # 另一个可用模型
}
def model_from_config(config, disable_tqdm=True):
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    elif model_type == "Llama_Forward":
        return Llama_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "Flan_T5":
        return Flan_T5(config, disable_tqdm=disable_tqdm)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def make_llm_client_from_config(gpt_cfg: dict):
    base = gpt_cfg.get("api_base") or gpt_cfg.get("base_url") or os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    key = gpt_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    model_name = gpt_cfg.get("model") or "openai/gpt-3.5-turbo"

    # 优先用新版 SDK + 官方 sample 那种风格 (需要有 OpenAI 类和 base+key)
    if base and key and HAS_NEW_OPENAI:
        try:
            client = OpenAI(base_url=base.rstrip("/"), api_key=key)
            return client, model_name  # new-style 允许 "openai/gpt-3.5-turbo"
        except Exception:
            pass  # 失败就退回到兼容 old-style

    # fallback 到老的 openai 兼容客户端
    import openai as _openai
    if key:
        _openai.api_key = key
    if base:
        _openai.api_base = base.rstrip("/")

    # 旧接口不识别 "openai/..." 前缀，去掉它
    if isinstance(model_name, str) and model_name.startswith("openai/"):
        model_name = model_name.split("/", 1)[1]

    return _openai, model_name


async def dispatch_chat_completions_async(
    client,
    messages_list: list[list[dict]],
    model: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
):
    calls = []
    for messages in messages_list:
        if HAS_NEW_OPENAI and hasattr(client, "chat") and hasattr(client.chat, "completions"):
            calls.append(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    extra_headers={},
                    extra_body={},
                )
            )
        else:
            # old openai interface
            calls.append(
                client.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            )
    return await asyncio.gather(*calls)
class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass

class Llama_Forward(LLM):
    """Wrapper for llama."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        SIZE=13
        MODEL_DIR = "/home2/langj/Covariates-improvement-in-Prompt-based-LLM/models/vicuna-13b"
        TOKENIZER_DIR = MODEL_DIR
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.model=LlamaForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
        self.tokenizer=LlamaTokenizer.from_pretrained(TOKENIZER_DIR)

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompts, n):
        if not isinstance(prompts, list):
            prompts = [prompts]
        text = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # Generate
            generate_ids = self.model.generate(input_ids, max_new_tokens=32)
            text.append(self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, " 
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens


class Flan_T5(LLM):
    """Wrapper for llama."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.device="cuda:1"
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl",
                                                    torch_dtype=torch.float16).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompts, n):
        if not isinstance(prompts, list):
            prompts = [prompts]
        # prompt_batches = [prompt[i:i + batch_size]
        #                   for i in range(0, len(prompt), batch_size)]
        # if not self.disable_tqdm:
        #     print(
        #         f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
        #         f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []
        batch_size=10
        for i in range(len(prompts) // batch_size):
            tmp_prompts = prompts[i*batch_size:(i+1)*batch_size]
            input_ids = self.tokenizer(tmp_prompts, padding='longest', return_tensors="pt").input_ids.to(device=self.device)
            outputs = self.model.generate(input_ids, max_new_tokens=32)
            text += self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            
        # batch_size = int(len(prompts)/2)
        # prompts1 = prompts[:batch_size]
        # prompts2 = prompts[batch_size:]
        # input_ids1 = self.tokenizer(prompts1, padding='longest', return_tensors="pt").input_ids.to(device=self.device)
        # input_ids2 = self.tokenizer(prompts2, padding='longest', return_tensors="pt").input_ids.to(device=self.device)
        # outputs1 = self.model.generate(input_ids1, max_new_tokens=20)
        # text += self.tokenizer.batch_decode(outputs1, skip_special_tokens=True)
        # outputs2 = self.model.generate(input_ids2, max_new_tokens=20)
        # text += self.tokenizer.batch_decode(outputs2, skip_special_tokens=True) 
        # batch_size=1        
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens





class GPT_Forward(LLM):
    """Wrapper for ChatGPT."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

        # 从 gpt_config 构造 client 和 model 名
        gpt_cfg = self.config.get("gpt_config", {}) or {}
        self.llm_client, self.model_name = make_llm_client_from_config(gpt_cfg)
        self.gpt_cfg = gpt_cfg  # 后面用来拿 temperature / max_tokens 等

    def chat_completion(self, messages, is_async=False):
        """
        Unified chat completion wrapper that handles new OpenAI SDK and fallback.
        `messages`:
          - if is_async=False: list of message dicts, e.g. [{"role":"user","content":"..."}]
          - if is_async=True: list of lists of messages (for batch)
        """
        temperature = self.gpt_cfg.get("temperature", 0.0)
        max_tokens = self.gpt_cfg.get("max_tokens", 256)
        frequency_penalty = self.gpt_cfg.get("frequency_penalty", 0.0)
        presence_penalty = self.gpt_cfg.get("presence_penalty", 0.0)

        if is_async:
            # messages is a list of message-lists
            responses = asyncio.run(
                dispatch_chat_completions_async(
                    client=self.llm_client,
                    messages_list=messages,
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            )
            return responses  # raw response objects
        else:
            # single synchronous call
            if HAS_NEW_OPENAI and hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    extra_headers={},
                    extra_body={},
                )
                try:
                    return completion.choices[0].message.content
                except Exception:
                    return completion.choices[0].get("message", {}).get("content", "")
            else:
                resp = self.llm_client.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                try:
                    return resp["choices"][0]["message"]["content"]
                except Exception:
                    return ""

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)


    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.__generate_text(prompt_batch, n)

        return text

    def complete(self, prompt, n):
        """用 chat-completion 生成文本（替代旧的 openai.Completion 接口）。"""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__generate_text(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
            """
            Chat-style 模型（如 gpt-3.5-turbo）不原生支持 token-level logprobs,
            这里返回空结构以满足接口（避免抽象类未实现报错）。
            """
            if not isinstance(text, list):
                text = [text]
            # 返回和输入等长的空列表（占位），调用方要能处理这种情况
            log_probs = [[] for _ in text]
            tokens = [[] for _ in text]
            return log_probs, tokens

    def __generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        answer = []
        for p in prompt:
            prompt_single = p.replace('[APE]', '').strip()
            content = self.chat_completion([{"role": "user", "content": prompt_single}], is_async=False)
            if content:
                answer.append(content)
            else:
                answer.append("do not have response from chatgpt")
        return answer

    def __async_generate(self, prompt, n):
        ml = [[{"role": "user", "content": p.replace('[APE]', '').strip()}] for p in prompt]
        answer = None
        while answer is None:
            try:
                predictions = self.chat_completion(ml, is_async=True)
            except Exception as e:
                print(e)
                print("Retrying....")
                time.sleep(20)
                continue
            answer = []
            for x in predictions:
                if HAS_NEW_OPENAI and hasattr(x, "choices"):
                    try:
                        content = x.choices[0].message.content
                    except Exception:
                        content = x.choices[0].get("message", {}).get("content", "")
                else:
                    # old style
                    try:
                        content = x["choices"][0]["message"]["content"]
                    except Exception:
                        content = x.get("choices", [{}])[0].get("message", {}).get("content", "")
                answer.append(content)
        return answer

        # try:          
    # reply = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": input}],
    #     temperature=0.0,
    #     max_tokens=args.max_length_cot,
    #     frequency_penalty=0,
    #     presence_penalty=0)
    # reply = reply['choices'][0]["message"]["content"].replace('\n\n', '')
    



    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index


class Claude_Forward(LLM):
    """Wrapper for Claude."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            # text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
            text += self.__async_generate(prompt_batch, n)
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        answer = []
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt_single = prompt[i].replace('[APE]', '').strip()
            response = None

            while response is None:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt_single}],
                        temperature=0.0,
                        max_tokens=256,
                        frequency_penalty=0,
                        presence_penalty=0)

                except Exception as e:
                    if 'is greater than the maximum' in str(e):
                        raise BatchSizeException()
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
                try:
                    # print(response['choices'][0]["message"]["content"])
                    answer.append(response['choices'][0]["message"]["content"])
                except Exception:
                    answer.append('do not have reponse from chatgpt')

        return answer
    
    def __async_generate(self, prompt, n):
        ml = [[{"role": "user", "content": p.replace('[APE]', '').strip()}] for p in prompt]
        answer = None

        while answer is None:
            try:
                predictions = asyncio.run(dispatch_openai_requests(
                    messages_list = ml,
                    model='gpt-3.5-turbo',
                    temperature=0,
                    max_tokens=256,
                    frequency_penalty=0,
                    presence_penalty=0
                    )
                )
            except Exception as e:
                # if 'is greater than the maximum' in str(e):
                #     raise BatchSizeException()
                print(e)
                print("Retrying....")
                time.sleep(20)

            try:
                answer = [x['choices'][0]['message']['content'] for x in predictions]
            except Exception:
                print("Please Wait!")

        return answer
        # try:          
    # reply = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": input}],
    #     temperature=0.0,
    #     max_tokens=args.max_length_cot,
    #     frequency_penalty=0,
    #     presence_penalty=0)
    # reply = reply['choices'][0]["message"]["content"].replace('\n\n', '')
    

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        while response is None:
            try:
                response = openai.Completion.create(
                    **config, prompt=prompt)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        return response['choices']

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if log_prob_range is not None:
            for i in range(len(text)):
                lower_index, upper_index = log_prob_range[i]
                assert lower_index < upper_index
                assert lower_index >= 0
                assert upper_index - 1 < len(text[i])
        config = self.config['gpt_config'].copy()
        config['logprobs'] = 1
        config['echo'] = True
        config['max_tokens'] = 0
        if isinstance(text, list):
            text = [f'\n{text[i]}' for i in range(len(text))]
        else:
            text = f'\n{text}'
        response = None
        while response is None:
            try:
                response = openai.Completion.create(
                    **config, prompt=text)
                # import pdb;pdb.set_trace()

            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        log_probs = [response['choices'][i]['logprobs']['token_logprobs'][1:]
                     for i in range(len(response['choices']))]
        tokens = [response['choices'][i]['logprobs']['tokens'][1:]
                  for i in range(len(response['choices']))]
        offsets = [response['choices'][i]['logprobs']['text_offset'][1:]
                   for i in range(len(response['choices']))]

        # Subtract 1 from the offsets to account for the newline
        for i in range(len(offsets)):
            offsets[i] = [offset - 1 for offset in offsets[i]]

        if log_prob_range is not None:
            # First, we need to find the indices of the tokens in the log probs
            # that correspond to the tokens in the log_prob_range
            for i in range(len(log_probs)):
                lower_index, upper_index = self.get_token_indices(
                    offsets[i], log_prob_range[i])
                log_probs[i] = log_probs[i][lower_index:upper_index]
                tokens[i] = tokens[i][lower_index:upper_index]

        return log_probs, tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index






class GPT_Insert(LLM):

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        assert batch_size == 1
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, split into {len(prompt_batches)} batches of (maximum) size {batch_size * n}")
        text = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def log_probs(self, text, log_prob_range=None):
        raise NotImplementedError

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # Split prompts into prefixes and suffixes with the [APE] token (do not include the [APE] token in the suffix)
        prefix = prompt[0].split('[APE]')[0]
        suffix = prompt[0].split('[APE]')[1]
        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    **config, prompt=prefix, suffix=suffix)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        # Remove suffix from the generated text
        texts = [response['choices'][i]['text'].replace(suffix, '') for i in range(len(response['choices']))]
        return texts
    
    


def gpt_get_estimated_cost(config, prompt, max_tokens):
    """更新为OpenRouter的定价"""
    prompt = prompt.replace('[APE]', '')
    n_prompt_tokens = len(prompt) // 4
    
    # OpenRouter定价（每千token）
    openrouter_costs = {
        'meta-llama/llama-3-70b-instruct': 0.00059,
        'microsoft/wizardlm-2-8x22b': 0.00065,
        # 添加其他你使用的模型
    }
    
    model_name = config['gpt_config']['model']
    cost_per_thousand = openrouter_costs.get(model_name, 0.001)  # 默认值
    
    total_tokens = n_prompt_tokens + max_tokens
    price = cost_per_thousand * total_tokens / 1000
    return price

class BatchSizeException(Exception):
    pass
