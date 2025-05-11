import json

input_file = r'data\synpt_Doubao-pro_Doubao-pro.jsonl'
output_file = r'train_data\synpt_Doubao-pro_Doubao-pro.json'

with open(input_file, 'r', encoding='utf-8') as f:
    input_list = [json.loads(line) for line in f]

output_list = []

for item in input_list:
    if item['state'] == False:
        continue
    item_list = []
    for message in item['messages']:
        item_dict = dict()
        if message['role'] == 'user':
            item_dict['role'] = message['role']
            item_dict['content'] = message['content']
        elif message['role'] == 'assistant':
            item_dict['role'] = message['role']
            content = ''
            if 'intention_thought' in message:
                content += f"用户意图思考：{message['intention_thought']}\n"
            if 'emotion_thought' in message:
                content += f"用户情感思考：{message['emotion_thought']}\n"
            if 'options_thought' in message:
                content += f"参考选项思考：{message['options_thought']}\n"
            
            content  = f"<think>[思考过程]：{content}</think>"

            if 'content' in message:
                content += f"<answer>[回复]：{message['content']}</answer>"
            if 'summary' in message:
                content += f"<summary>[用户意图总结]：{message['summary']}</summary>"
            item_dict['content'] = content
        else:
            continue 
        item_list.append(item_dict)
    output_list.append({'messages': item_list})
print(f"共有{len(output_list)}条数据")

with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_list, json_file, ensure_ascii=False, indent=4)