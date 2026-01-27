"""
Here is the Chinese version of `search_for_complex_reasoning_path.py`.  
By using it, it will generate reasoning paths in Chinese, along with the thought process and responses in Chinese.  
If you need to generate data in English, please use the original `search_for_complex_reasoning_path.py`.
"""

import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy

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

verify_prompt = """<模型回答>
{}
</模型回答>

<参考答案>
{}
</参考答案>

你将获得一段模型生成的回答<模型回答>以及对应的参考答案<参考答案>。
请比较两者并判断模型回答是否正确。
你的任务就只输出：
- "True"（正确）
- "False"（不正确）"""


query_prompt_init = """<问题>
{}
</问题>

请使用链式思维（CoT）方法回答以上 <问题> 中的问题。你的回复应包含多个步骤，每个步骤包含三类动作：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

- **'Inner Thinking'**：进行思考的步骤。需要多个 'Inner Thinking' 以展示完整推理。每一步首先生成一个简短标题。
- **'Final Conclusion'**：总结前面 'Inner Thinking' 的正确推理并给出最终答案。此处不需要标题。
- **'Verification'**：核验 "Final Conclusion" 的结论是否成立。若成立则结束；否则返回 "Inner Thinking" 继续推理。此处不需要标题。

输出格式必须严格遵循下面的 JSON 结构，且 JSON 字段内所有内容必须使用**中文**：
```json
{{
  "CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
  ]
}}
```"""

gen_prompt_rethink_Backtracking = """<问题>
{}
</问题>

<先前推理>
{}
</先前推理>

<回复要求>
你的回复必须包含以下步骤，每个步骤由三类动作构成：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

1. **"Inner Thinking"**：将推理过程拆解为多个简洁步骤。每一步以一个简短标题（title）开头，说明该步目的。
2. **"Final Conclusion"**：汇总所有 "Inner Thinking" 的正确推理并给出最终答案。本步不需要 title。
3. **"Verification"**：核验 "Final Conclusion" 的正确性。若成立则结束；否则回到 "Inner Thinking" 继续完善。本步不需要 title。

</回复要求>

其中，<问题> 表示要回答的问题，<先前推理> 包含你之前的推理过程。你的任务是从当前的 "Verification" 步继续。
我已人工审阅并确认：你之前的 **"Final Conclusion" 是错误的**，因此你接下来的 "Verification" 结论必须与此一致。

请使用**回溯（backtracking）**策略，回到较早的推理节点，重新构建一条正确路径，并给出新的 "Final Conclusion"。

### 输出格式
严格遵循以下 JSON 结构。JSON 字段内所有内容必须使用**中文**。不需要重复粘贴你之前的推理，从下一步 "Verification" 直接开始。

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Exploring_New_Path = """<问题>
{}
</问题>

<先前推理>
{}
</先前推理>

<回复要求>
你的回复必须包含以下步骤，每个步骤由三类动作构成：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

1. **"Inner Thinking"**：将推理过程拆解为多个简洁步骤。每一步以一个简短标题（title）开头，说明该步目的。
2. **"Final Conclusion"**：汇总所有 "Inner Thinking" 的正确推理并给出最终答案。本步不需要 title。
3. **"Verification"**：核验 "Final Conclusion" 的正确性。若成立则结束；否则回到 "Inner Thinking" 继续完善。本步不需要 title。

</回复要求>

其中，<问题> 表示要回答的问题，<先前推理> 包含你之前的推理过程。你的任务是从当前的 "Verification" 步继续。
我已人工审阅并确认：你之前的 **"Final Conclusion" 是错误的**，因此你接下来的 "Verification" 结论必须与此一致。

请使用**探索新路径（exploring new approaches）**策略，尝试不同的解题思路，构建新的推理链并给出新的 "Final Conclusion"。

### 输出格式
严格遵循以下 JSON 结构。JSON 字段内所有内容必须使用**中文**。不需要重复粘贴你之前的推理，从下一步 "Verification" 直接开始。

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Verification = """<问题>
{}
</问题>

<先前推理>
{}
</先前推理>

<回复要求>
你的回复必须包含以下步骤，每个步骤由三类动作构成：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

1. **"Inner Thinking"**：将推理过程拆解为多个简洁步骤。每一步以一个简短标题（title）开头，说明该步目的。
2. **"Final Conclusion"**：汇总所有 "Inner Thinking" 的正确推理并给出最终答案。本步不需要 title。
3. **"Verification"**：核验 "Final Conclusion" 的正确性。若成立则结束；否则回到 "Inner Thinking" 继续完善。本步不需要 title。

</回复要求>

其中，<问题> 表示要回答的问题，<先前推理> 包含你之前的推理过程。你的任务是从当前的 "Verification" 步继续。
我已人工审阅并确认：你之前的 **"Final Conclusion" 是错误的**，因此你接下来的 "Verification" 结论必须与此一致。

请使用**强化验证（validation/verification）**策略，系统检查先前推理中的薄弱环节，补全关键证据，并给出新的 "Final Conclusion"。

### 输出格式
严格遵循以下 JSON 结构。JSON 字段内所有内容必须使用**中文**。不需要重复粘贴你之前的推理，从下一步 "Verification" 直接开始。

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<问题>
{}
</问题>

<先前推理>
{}
</先前推理>

<回复要求>
你的回复必须包含以下步骤，每个步骤由三类动作构成：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

1. **"Inner Thinking"**：将推理过程拆解为多个简洁步骤。每一步以一个简短标题（title）开头，说明该步目的。
2. **"Final Conclusion"**：汇总所有 "Inner Thinking" 的正确推理并给出最终答案。本步不需要 title。
3. **"Verification"**：核验 "Final Conclusion" 的正确性。若成立则结束；否则回到 "Inner Thinking" 继续完善。本步不需要 title。

</回复要求>

其中，<问题> 表示要回答的问题，<先前推理> 包含你之前的推理过程。你的任务是从当前的 "Verification" 步继续。
我已人工审阅并确认：你之前的 **"Final Conclusion" 是错误的**，因此你接下来的 "Verification" 结论必须与此一致。

请使用**纠错（correction）**策略：明确指出先前结论错误的关键点，修正错误假设或推导，并给出新的 "Final Conclusion"。

### 输出格式
严格遵循以下 JSON 结构。JSON 字段内所有内容必须使用**中文**。不需要重复粘贴你之前的推理，从下一步 "Verification" 直接开始。

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_w_label = """<问题>
{}
</问题>

<先前推理>
{}
</先前推理>

<回复要求>
你的回复必须包含以下步骤，每个步骤由三类动作构成：**"Inner Thinking"**、**"Final Conclusion"**、**"Verification"**：

1. **"Inner Thinking"**：将推理过程拆解为多个简洁步骤。每一步以一个简短标题（title）开头，说明该步目的。
2. **"Final Conclusion"**：汇总所有 "Inner Thinking" 的正确推理并给出最终答案。本步不需要 title。
3. **"Verification"**：核验 "Final Conclusion" 的正确性。若成立则结束；否则回到 "Inner Thinking" 继续完善。本步不需要 title。

</回复要求>

其中，<问题> 表示要回答的问题，<先前推理> 包含你之前的推理过程。你的任务是从当前的 "Verification" 步继续。

现在我将“偷偷告诉你”标注答案是“{}”，但你必须假装并不知道。
你的 "Verification" 需要认真权衡；如果发现不一致，你必须给出新的 "Inner Thinking" 与新的 "Final Conclusion"，确保最终答案与正确标注对齐。

### 输出格式
严格遵循以下 JSON 结构。JSON 字段内所有内容必须使用**中文**。不需要重复粘贴你之前的推理，从下一步 "Verification" 直接开始。

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

reformat_to_complex_cot_prompt = """<思考过程>
{}
</思考过程>

<问题>
{}
</问题>

以上 <思考过程> 反映了模型基于 <问题> 的推理路径。请将该 <思考过程> 改写为更贴近人类、直觉化的中文自然思考过程，要求：

1. 以逐步推理的形式呈现，**每一条想法单独成行，用换行符分隔**。
2. 避免结构化标题或刻板格式，强调自然衔接。
3. 可以使用更口语、更自然的过渡或自检表达，例如“嗯”、“等等”、“另外”、“我再确认一下”等。
4. 适当扩写，使推理更丰富、更细致、更合乎逻辑，但仍保持对话式与直觉化。
5. 必须至少包含 5 行，每行用换行符 \n 分隔；不得输出成单段话。

请直接以 JSON 形式返回改写后的自然思考过程：
```json
{{
  "NaturalReasoning": "..."
}}
```
"""

get_final_response_prompt = """<内部思考>
{}
</内部思考>

<问题>
{}
</问题>

<内部思考> 表示你对 <问题> 的内部推理。请基于此生成给用户的最终回答（中文），要求：
- 如果存在明确答案，请在开头先给出答案；
- 回答应紧扣 <问题>；
- 只输出最终回答正文，不要输出任何额外内容。
"""
# search strategies
search_strategies = [('Backtracking',gen_prompt_rethink_Backtracking),('Exploring New Paths',gen_prompt_rethink_Exploring_New_Path),('Verification',gen_prompt_rethink_Verification),('Correction',gen_prompt_rethink_Correction)]



def extract_bracket_content(text):
        # Extract content between the first '{' and the last '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

def parse_gpt_response(response):
    try:
        if not response or not isinstance(response, str):
            raise ValueError("empty response")
        #判断响应字符串的第一个字符是否为左大括号 ‘{’（JSON 的标准起始符）
        if '{' != response[0]:
            response = extract_bracket_content(response)
        #json.loads(...)：将JSON 字符串，解析为 Python 字典 da；
        da = json.loads(response.replace('\n',''))
        assert isinstance(da["CoT"],list), "CoT should be list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None

def parse_gpt_response_reformat(response):
    try:
        if not response or not isinstance(response, str):
            raise ValueError("empty response")

        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))
       
        assert isinstance(da["NaturalReasoning"],str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have \\n"
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None 
    
#将结构化的列表数据格式化为统一的 Markdown 风格文本串
def get_stream_of_search(longcot):
    """
    temp = '### {}\n{}\n'：固定文本格式，用 ### 标记二级标题（Markdown 语法），结构为：
    第一行：### 标题内容（对应字典中的某个字段）
    第二行：内容详情（对应字典中的 content 字段）
    末尾换行，保证多条记录分隔清晰
    """
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'],x['content']))
        else:
            resstr.append(temp.format(x['action'].replace('Final Conclusion','Conclusion'),x['content']))
    return '\n'.join(resstr).strip() #用换行符 \n 连接，最后用 strip() 去除首尾可能的多余空白

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--max_search_attempts", type=int, default=1, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth.")
    parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    
    args = parser.parse_args()

    def filter_data(tmpdata):
        filtered_data = []
        for da in tmpdata:
            if '开放式可验证问题' not in da or '标准答案' not in da:
                continue
            filtered_data.append(da)

        print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
        return filtered_data

    with open(args.data_path) as f:
        #自动解析 JSON 字符串，转换为对应 Python 类型（如 JSON 对象→字典、JSON 数组→列表、数字 / 字符串保持对应类型）
        tmpdata = json.load(f)

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    if args.limit_num:
        data = data[:args.limit_num]
        
    print(f"read data:{len(data)}")

    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_CoT_search_{args.model_name}'
    save_dir = f'output_data/{task_name}'

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)


    def verify_gpt(conclusion,answer,d):
        query = verify_prompt.format(conclusion,answer)
        response = gpt_instance.retry_call(query)
        d['qwen_query_cot'].append(query)
        d['qwen_response_cot'].append(response)
        if 'true' in response.lower():
            d['verify'].append(True)
            return True
        else:
            d['verify'].append(False)
            return False
        
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 2
            d['verify'] = []
            d['Long_CoT'] = []
            d['qwen_query_cot'] = []
            d['qwen_response_cot'] = []
            d['response_struct'] = []
            d['response_type'] = []
            d['prior_fail_try'] = []

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            # init reason
            query = query_prompt_init.format(d['开放式可验证问题'])
            d['qwen_query_cot'].append(query)
            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                if ii == 0:
                    d['qwen_response_cot'].append(response)
                flag, struct = parse_gpt_response(response)
                if flag:
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] =  struct["CoT"]
                    d['response_type'].append('Init_CoT')
                    break
                else:
                    print(f'retrying Init_CoT',flush=True)
            if not flag:
                raise Exception('init error')

            verify_gpt(d['Long_CoT'][-2]['content'],d['标准答案'],d)

            for rethinking_try_time in range(args.max_search_attempts):
                if rethinking_try_time > 0:
                    # Archive the failed state
                    del d['prior_fail_try'] #删除d中存储的 “历史失败记录” 字段（避免归档时重复嵌套）
                    save_d['prior_fail_try'].append(d)
                    # Replace with a new state
                    d = save_d

                # Save the initial state
                save_d = copy.deepcopy(d)

                # Begin search
                for rethink_time in range(args.max_search_depth):
                    if d['verify'][-1]:
                        break
                    """
                    json.dumps(...)：Python json 模块的序列化函数，将 Python 对象（此处是切片后的列表）转为 JSON 字符串；
                    ensure_ascii=False：保留非 ASCII 字符（如中文、特殊符号），不进行 ASCII 编码；
                    indent=2：JSON 字符串按 2 个空格缩进格式化
                    """
                    reasoning = json.dumps(d['Long_CoT'][:-1],ensure_ascii=False,indent=2)
                    # Search strategy
                    if rethink_time > 0:
                        strategy_name,strategy = random.choice(search_strategies)
                    else:
                        # exclude Backtracking ,第一次不选 Backtracking
                        strategy_name,strategy = random.choice(search_strategies[1:])

                    query = strategy.format(d['开放式可验证问题'],reasoning)
                    d['qwen_query_cot'].append(query)
                    
                    for ii in range(retry_time):
                        response = gpt_instance.retry_call(query)
                        flag, struct = parse_gpt_response(response)

                        if flag:
                            d['qwen_response_cot'].append(response)
                            d['response_struct'].append(struct["CoT"])
                            d['Long_CoT'] =  d['Long_CoT'][:-1] + struct["CoT"]
                            d['response_type'].append(f'Re_CoT_{strategy_name}')
                            break
                        else:
                            print(f'retrying strategy {strategy_name}',flush=True)
                    if not flag:
                        raise Exception('rethink error')
                    verify_gpt(d['Long_CoT'][-2]['content'],d['标准答案'],d)
                
                if d['verify'][-1]:
                    break

            # If it is still incorrect, generate a final Label_CoT round
            if not d['verify'][-1] and args.efficient_search:
                reasoning = json.dumps(d['Long_CoT'][:-1],ensure_ascii=False,indent=2)
                query = gen_prompt_w_label.format(d['开放式可验证问题'],reasoning,d['标准答案'])
                d['qwen_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)       
                    flag, struct = parse_gpt_response(response)
                    if flag:
                        d['qwen_response_cot'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] =  d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append('Label_CoT')
                        # ignore verify
                        d['verify'].append(True)
                        break
                    else:
                        print(f'retrying Label_CoT',flush=True)
                if not flag:
                    raise Exception('label error') 
            
            if d['verify'][-1]:
                # Generate complex CoT and final response (Complex_CoT, response)
                sos = get_stream_of_search(d['Long_CoT'])
                query = reformat_to_complex_cot_prompt.format(sos,d['开放式可验证问题'])
                d['qwen_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response_reformat(response)
                    if flag:
                        d['qwen_response_cot'].append(response)
                        d["Complex_CoT"] = struct["NaturalReasoning"]
                        # get response
                        query = get_final_response_prompt.format(d['Complex_CoT'],d['开放式可验证问题'])
                        d['qwen_query_cot'].append(query)
                        response = gpt_instance.retry_call(query)
                        d['qwen_response_cot'].append(response)
                        d["Response"] = response
                        d['Question'] = d['开放式可验证问题']
                        break
                    
            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False,indent=2)
                wrongtime = 0

        except Exception as e:
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 20:
                assert 1 == 0, 'wrong'
        return 1
            
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]


    def merge_saved_files(save_dir):
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json')]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'Complex_CoT' in da and 'Response' in da
                    res.append(da)
            except Exception as e:
                continue
        return res
    
    os.makedirs(save_dir, exist_ok=True)

    # Merge previously processed files
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, input_data), total=len(input_data), desc="Processing samples", unit="sample"))

     # Merge and save final output
    final_data = merge_saved_files(save_dir)
    output_path = f"{task_name}_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()