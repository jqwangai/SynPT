import random
import time
import asyncio
import aiohttp
import numpy as np

def from_message_to_text(messages):
    text = ''
    for message in messages:
        if message['role'] == 'user':
            text += f"用户: {message['content']}\n"
        elif message['role'] == 'assistant':
            text += f"文旅助手: {message['content']}\n"
        else:
            text += f"{message['role']}: {message['content']}\n"
    return text

def from_message_to_text_current_turn(messages):
    text = ''
    if len(messages) == 1:
        return f"用户: {messages[0]['content']}\n"
    for message in messages[-2:]:
        if message['role'] == 'user':
            text += f"用户: {message['content']}\n"
        elif message['role'] == 'assistant':
            text += f"文旅助手: {message['content']}\n"
        else:
            text += f"{message['role']}: {message['content']}\n"
    return text

def get_info_from_config(task_configs, data):

    task_config = task_configs[data['category']]
    intention_info = ''
    for intention_id, intention_type in zip(task_config['columns'], task_config['columns_description']):
        intention_info += f"{intention_type}：{data[intention_id]}\n"
    return task_config['task_name'], task_config['columns_description'], intention_info, task_config['max_turns']

def get_random_intention_from_config(task_configs,data):
    task_config = task_configs[data['category']]
    intention_info_selected = ''
    for column, column_des, column_prob in zip(task_config['columns'], task_config['columns_description'],task_config['keys_prob']):
        if random.random() < column_prob:
            intention_info_selected += f"{column_des}：{data[column]}\n"
    return intention_info_selected, task_config['task_name']

from scipy.stats import norm
def get_random_intention_from_config_prob(task_configs,data):
    task_config = task_configs[data['category']]
    # 概率采样意图个数
    intention_nums = np.arange(0, len(task_config['columns']) + 1)
    if task_config['mean'] == -1:
        mean = len(intention_nums)/ 2
    else:
        mean = float(task_config['mean'])
    if task_config['std'] == -1:
        std = 2
    else:
        std = float(task_config['std'])
    probability_density = norm.pdf(intention_nums, mean, std)
    weights = probability_density / probability_density.sum()
    random.seed(int(data['id']))
    selected_number = random.choices(intention_nums, weights=weights, k=1)[0]

    # 根据概率采样得到意图个数
    keys_prob = [prob / sum(task_config['keys_prob']) for prob in task_config['keys_prob']]
    random.seed(int(data['id']))
    indices = np.random.choice(a=range(len(task_config['columns'])), size=selected_number, replace=False, p=keys_prob)
    indices.sort()
    intention_info_selected = ''
    for i in indices:
        intention_info_selected += f"{task_config['columns_description'][i]}：{data[task_config['columns'][i]]}\n"
    return intention_info_selected, task_config['task_name']

def get_simple_messages(system_prompt, user_content):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

def call_openai_api(client, model, messages, temperature=0.01, max_retries=5, backoff_factor=1):
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=messages,
                temperature = temperature
            )
            model_response = response.choices[0].message.content
            return model_response
        except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
                if attempt < max_retries:
                    print(f"Retrying in {backoff_factor} seconds...")
                    time.sleep(backoff_factor)  # 延时重试
                else:
                    return f"Error: {str(e)}"
def call_openai_api_token_usage(client, model, messages, temperature=0.01, max_retries=5, backoff_factor=1):
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=messages,
                temperature = temperature
            )
            model_response = response.choices[0].message.content
            return model_response, {'prompt_tokens':response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens}
        except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
                if attempt < max_retries:
                    print(f"Retrying in {backoff_factor} seconds...")
                    time.sleep(backoff_factor)  # 延时重试
                else:
                    return f"Error: {str(e)}"

async def call_openai_api_token_usage_async(client, model, messages, temperature=0.01, max_tokens=2048, max_retries=5, backoff_factor=1):
    attempt = 0
    url = f"{client.base_url}chat/completions"
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens":max_tokens
    }
    
    async with aiohttp.ClientSession() as session:
        while attempt < max_retries:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_response = data['choices'][0]['message']['content']
                        usage_info = data['usage']
                        return model_response, {'prompt_tokens': usage_info['prompt_tokens'], 'completion_tokens': usage_info['completion_tokens'], 'total_tokens': usage_info['total_tokens']}
                    else:
                        error_message = await response.text()
                        raise Exception(f"Error {response.status}: {error_message}")
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
                if attempt < max_retries:
                    print(f"Retrying in {backoff_factor} seconds...")
                    await asyncio.sleep(backoff_factor)  # 异步延时重试
                else:
                    return f"Error: {str(e)}"

def count_tokens(token_usage, current_usage):
    for key in token_usage.keys():
        token_usage[key] += current_usage[key]
                