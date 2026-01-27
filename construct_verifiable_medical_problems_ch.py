import os
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
import argparse
import traceback
import re
import requests

class GPT:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **additional_args,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data['choices'][0]['message']['content']

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--filter_data", action='store_true', help="Enable filtering of questions with LLMs.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--num_process", type=int, default=10, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    return parser.parse_args()

def extract_bracket_content(text):
    # Extract content between the first '{' and the last '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def load_input_data(data_path: str):
    """
    支持两种输入：
    - .json   : 一个 JSON 数组（list[dict]）
    - .jsonl  : JSON Lines，每行一个 dict
    """
    _, ext = os.path.splitext(data_path)
    ext = ext.lower()

    if ext == ".jsonl":
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"JSONL 解析失败: {data_path}:{line_no} -> {e}")
        return data

    # 默认按 .json 数组读取
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"期望 {data_path} 是 JSON 数组(list)，但实际是: {type(obj)}")
    return obj

def parse_gpt_response(response):
    try:
        if not response.startswith('{'):
            response = extract_bracket_content(response)
        parsed_data = json.loads(response.replace('\n', ''))

        # # 1. 自动修正 "Question" 相关的 Key 拼写错误
        # # 找到所有可能的 key 变体
        # question_key_candidates = [
        #     "Open-ended Verifiable Question", # 正确的
        #     "Open-ended Verolvable Question", # 你遇到的错误
        #     "Open-ended Question",            # 常见的简化
        #     "Question", "question"            # 最基础的
        # ]
        
        # # 遍历候选列表，谁在字典里就用谁的值，并赋给标准 Key
        # found_q = False
        # for key in question_key_candidates:
        #     if key in parsed_data:
        #         parsed_data["Open-ended Verifiable Question"] = parsed_data[key] # 统一归位
        #         found_q = True
        #         break
        
        # if not found_q:
        #     # 如果实在找不到，可以尝试拿字典的第一个值当作 Question（终极兜底）
        #     pass 

        # # 2. 自动修正 "Answer" 相关的 Key 拼写错误 (防止 Answer 也拼错)
        # answer_key_candidates = [
        #     "Ground-True Answer",  # 正确的
        #     "Ground-Truth Answer", # 常见的变体 (True vs Truth)
        #     "Answer", "answer"
        # ]
        # for key in answer_key_candidates:
        #     if key in parsed_data:
        #         parsed_data["Ground-True Answer"] = parsed_data[key]
        #         break

        assert len(parsed_data) == 2, "Response JSON should contain exactly two keys."
        assert isinstance(parsed_data["开放式可验证问题"], str), "开放式可验证问题必须是字符串。"
        assert isinstance(parsed_data["标准答案"], str), "标准答案必须是字符串。"

        return True, parsed_data
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        print(f"\n[DEBUG] Parsing Failed!") 
        print(f"[DEBUG] Error Type: {e}")
        print(f"[DEBUG] Model Raw Response: {response}\n")
        return False, None

def process_single_item(item, gpt_instance, save_directory, filter_prompt, reformat_prompt, filter_enabled):
    try:
        max_retries = 2
        save_path = os.path.join(save_directory, f"{item['process_id']}.json")

        # Generate options string for the question
        item['options_str'] = '\n'.join([f"{key}. {value}" for key, value in item['options'].items()])
        question_text = f"{item['question']}\n{item['options_str']}"

        # Filter questions if enabled
        if filter_enabled:
            filter_query = filter_prompt.format(question_text, item['answer'])
            item['gpt_filter_query'] = filter_query
            response = gpt_instance.retry_call(filter_query)
            item['gpt_filter_response'] = response

            if "通过" not in (response or ""):
                with open(save_path, 'w', encoding='utf-8') as file:
                    json.dump(item, file, ensure_ascii=False, indent=2)
                return 1

        # Reformat questions into open-ended format
        reformat_query = reformat_prompt.format(question_text, item['answer'])
        item['gpt_reformat_query'] = reformat_query

        for _ in range(max_retries):
            response = gpt_instance.retry_call(reformat_query)
            item['gpt_reformat_response'] = response
            valid, parsed_data = parse_gpt_response(response)

            if valid:
                item["开放式可验证问题"] = parsed_data["开放式可验证问题"]
                item["标准答案"] = parsed_data["标准答案"]
                break

        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing item {item['process_id']}: {e}")
    return 1

def merge_saved_files(directory):
    _, _, filenames = next(os.walk(directory))
    json_files = [f for f in filenames if f.endswith('.json')]
    merged_data = []

    for file in json_files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert (
                    '开放式可验证问题' in data
                    or 'gpt_filter_response' in data
                    or 'gpt4_response_filter' in data
                )
                # assert 'Open-ended Verifiable Question' in data or 'gpt_filter_response' in data  or 'gpt4_response_filter' in data
                merged_data.append(data)
        except Exception as e:
            # traceback.print_exc()
            print(f"Error merging file {file}: {e}")
    return merged_data

def deduplicate_data(data, processed_data):
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]

def main():
    args = parse_arguments()

    input_data = load_input_data(args.data_path)

    # # Load input data
    # with open(args.data_path, 'r') as file:
    #     input_data = json.load(file)

    # Assign unique process IDs to each item
    for idx, item in enumerate(input_data, start=1):
        item['process_id'] = idx

    if args.limit_num:
        input_data = input_data[:args.limit_num]

    print(f"Loaded {len(input_data)} items.")

    # Define task and save directory
    task_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_directory = os.path.join('output_data', task_name)
    os.makedirs(save_directory, exist_ok=True)

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)

    filter_prompt = """<选择题>
{}
正确答案：{}
</选择题>

你是一名擅长筛选与评估高质量推理题的专家。请判断该选择题是否满足以下条件：
1. **推理深度**：题目需要一定推理深度；如果过于简单，标记为“太简单”。
2. **答案唯一且无歧义**：必须存在唯一明确的正确答案；如果是“选错误项”、或可能多解，标记为“答案有歧义”。
3. **可改写为开放式问答**：应当能够改写成不带选项的开放问题，并且有清晰可核验的标准答案；否则标记为“不可改写”。

请仅输出以下之一：
- “通过”
- “太简单”
- “答案有歧义”
- “不可改写”
"""

    reformat_prompt = """我将提供一道选择题。你的任务是将其改写为一个不包含选项的开放式问题，并给出“标准答案”。要求如下：

1. 问题必须具体，准确命中原选择题所考察的知识点；不提供选项，但必须存在唯一明确的标准答案。
2. 标准答案要尽量简洁，便于字符串匹配/核验。

下面是需要改写的选择题：
<选择题>
{}
正确答案：{}
</选择题>

请严格输出一个合法 JSON 对象，且**只能**包含以下两个键（键名必须完全一致）：
1. "开放式可验证问题"
2. "标准答案"
请按照以下json格式输出结果：
```json
{{
"开放式可验证问题":"...",
"标准答案":"..."
}}
```
"""

    # Merge previously processed files
    processed_data = merge_saved_files(save_directory)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(input_data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    # Process data using a thread pool
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(lambda item: process_single_item(item, gpt_instance, save_directory, filter_prompt, reformat_prompt, args.filter_data), input_data), total=len(input_data), desc="Processing Items", unit="item"))

    # Merge and save final output
    final_data = merge_saved_files(save_directory)
    output_path = f"{task_name}_final_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
