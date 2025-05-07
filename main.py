import argparse
import asyncio
import json
import random
import os
from typing import Dict, Any

import yaml
from tqdm import tqdm
from openai import OpenAI

from prompts import first_inquiry_system_prompt, user_prompt, user_prompt_anxiety
from utils import (
    get_info_from_config,
    get_random_intention_from_config_prob,
    get_simple_messages,
    from_message_to_text,
    call_openai_api_token_usage_async,
    count_tokens
)
from user import User
from assistant_agent import AssistantAgent

parser = argparse.ArgumentParser(description="Run dialogue simulation with LLMs.")
parser.add_argument('--input_file', type=str, default='data/seed_data.jsonl', help='Seed data file (jsonl)')
parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to config.yaml')
parser.add_argument('--output_file', type=str, help='Output file path (jsonl)')
parser.add_argument('--batch_size', type=int,  default=5, help='Number of items per batch')
parser.add_argument('--user_model_id', type=str, default='', help='User model ID')
parser.add_argument('--user_model_name', type=str,  default='Doubao-pro', help='User model display name')
parser.add_argument('--assistant_model_id', type=str,  default='', help='Assistant model ID')
parser.add_argument('--assistant_model_name', type=str,  default='Doubao-pro', help='Assistant model display name')
parser.add_argument('--user_api_key', type=str,  default = os.environ.get('DOUBAO_API_KEY'), help='API key for user model')
parser.add_argument('--assistant_api_key', type=str, default = os.environ.get('DOUBAO_API_KEY'), help='API key for assistant model')
parser.add_argument('--user_base_url', type=str, default="https://ark.cn-beijing.volces.com/api/v3", help='Base URL for user model')
parser.add_argument('--assistant_base_url', type=str, default="https://ark.cn-beijing.volces.com/api/v3", help='Base URL for assistant model')

async def process_dialogue(item: Dict[str, Any], output_file: str, config: Dict[str, Any],
                           user_client, assistant_client,
                           user_model_id: str, user_model_name: str,
                           assistant_model_id: str, assistant_model_name: str) -> None:
    """Process a single dialogue instance."""
    random.seed(int(item['id']))
    anxiety_switch = random.random() < 0.33
    token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    # First inquiry (initial assistant response)
    intention_info, task_name = get_random_intention_from_config_prob(config['task'], item)
    system_prompt = first_inquiry_system_prompt.format(task_name=task_name)
    user_content = f"用户意图信息：\n{intention_info}"
    messages = get_simple_messages(system_prompt, user_content)

    first_inquiry, usage = await call_openai_api_token_usage_async(user_client, user_model_id, messages)
    count_tokens(token_usage, usage)

    if first_inquiry is None:
        print(f'{item["id"]}: First inquiry is empty. Skipping this instance.')
        return

    # print('Round 1 dialogue')
    task_name, intention_types, intention_info, max_turns = get_info_from_config(config['task'], item)

    user_input_template = user_prompt_anxiety if anxiety_switch else user_prompt
    user_input = user_input_template.format(task_name=task_name, intention_info=intention_info)

    user = User('user', user_input, user_model_id, 0.01, user_client)
    assistant = AssistantAgent(assistant_client, assistant_model_id, task_name, intention_types)

    user_response = first_inquiry
    user_intention = None

    for i in range(max_turns):
        response_dict, user_intention, usage = await assistant.chat(user_response, user_intention)
        count_tokens(token_usage, usage)

        if response_dict is None:
            print(f'{item["id"]}: Assistant returned None. Ending dialogue.')
            break

        if 'summary' in response_dict:
            save_result(
                output_file, item, assistant.dialog_history_details, token_usage,
                completed=True, anxiety=anxiety_switch,
                user_model_name=user_model_name,
                assistant_model_name=assistant_model_name
            )
            break

        if i == max_turns - 1:
            print(f'{item["id"]}: Reached max turn limit. Ending dialogue.')
            save_result(
                output_file, item, assistant.dialog_history_details, token_usage,
                completed=False, anxiety=anxiety_switch,
                user_model_name=user_model_name,
                assistant_model_name=assistant_model_name
            )
            break

        # print(f'Round {i + 2} dialogue')
        history_text = from_message_to_text(assistant.dialog_history)
        user_prompt_text = (
            f'```历史对话：\n{history_text}```\n'
            f'根据历史对话、你的意图信息列表和角色定义，回复文旅助手的最后回答。'
        )
        user_response, usage = await user.chat(user_prompt_text)
        count_tokens(token_usage, usage)


def save_result(
    output_file: str,
    item: Dict[str, Any],
    messages: Any,
    token_usage: Dict[str, int],
    completed: bool,
    anxiety: bool,
    user_model_name: str,
    assistant_model_name: str
) -> None:
    """Save the result to a JSONL file."""
    result = {
        'state': completed,
        'model': f"user:{user_model_name}, assistant:{assistant_model_name}",
        'id': item['id'],
        'category': item['category'],
        'anxiety': anxiety,
        'messages': messages,
        'prompt_tokens': token_usage['prompt_tokens'],
        'completion_tokens': token_usage['completion_tokens'],
        'total_tokens': token_usage['total_tokens']
    }
    with open(output_file, 'a+', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False) + '\n')


async def process_batch(batch: list, output_file: str, config: Dict[str, Any], **kwargs) -> None:
    """Process a batch of items concurrently."""
    tasks = [
        asyncio.create_task(process_dialogue(item, output_file, config, **kwargs))
        for item in batch
    ]
    await asyncio.gather(*tasks)


async def main():
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"data/synpct_{args.user_model_name}_{args.assistant_model_name}.jsonl"

    # Initialize model clients
    user_client = OpenAI(api_key=args.user_api_key, base_url=args.user_base_url)
    assistant_client = OpenAI(api_key=args.assistant_api_key, base_url=args.assistant_base_url)

    # Read input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_list = [json.loads(line) for line in f]
    print(f"Loaded {len(input_list)} records successfully.")

    # Read YAML config
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for i in tqdm(range(0, len(input_list), args.batch_size)):
        batch = input_list[i:i + args.batch_size]
        await process_batch(
            batch, args.output_file, config,
            user_client=user_client,
            assistant_client=assistant_client,
            user_model_id=args.user_model_id,
            user_model_name=args.user_model_name,
            assistant_model_id=args.assistant_model_id,
            assistant_model_name=args.assistant_model_name
        )

if __name__ == '__main__':
    asyncio.run(main())
