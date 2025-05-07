from prompts import assistant_prompt_intention, assistant_prompt_summary, assistant_prompt_emotion, assistant_prompt_options, assistant_prompt_response
from utils import get_simple_messages, call_openai_api_token_usage_async, from_message_to_text_current_turn,count_tokens
import re

class AssistantAgent:
    def __init__(self, client, model, task_name, intention_types):
        self.model = model
        self.task_name = task_name
        self.intention_types = intention_types
        self.client = client
        self.dialog_history = []
        self.dialog_history_details = []
    def update_history(self, user_message, model_response):
        """
        Update the dialogue history.
        :param user_message: User input
        :param model_response: Model response
        """
        if user_message is not None:
            self.dialog_history.append({"role": "user", "content": user_message})
        if model_response is not None:
            self.dialog_history.append({"role": "assistant", "content": model_response})
    
    def update_history_details(self, user_message, model_response):
        """
        Update the detailed dialogue history.
        :param user_message: User input
        :param model_response: Model response
        """
        if user_message is not None:
            self.dialog_history_details.append({"role": "user", "content": user_message})
        if model_response is not None:
            temp_dict = dict()
            temp_dict['role'] = 'assistant'
            for k in model_response.keys():
                temp_dict[k] = model_response[k]
            self.dialog_history_details.append(temp_dict)
            
    async def chat(self, user_response, user_intention):
        # Count tokens
        current_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.update_history(user_response, None)
        current_history = from_message_to_text_current_turn(self.dialog_history)
        assistant_response = dict()

        # 1. User intention thinking
        thought_content = f"```本轮对话：\n{current_history}```\n```上一状态的用户意图：{user_intention}```\n根据以上信息，完成你的任务。"
        messages = get_simple_messages(assistant_prompt_intention.format(task_name=self.task_name, intention_types=self.intention_types), thought_content)
        intention_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(intention_response)

        # Parse intention_response
        user_intention_pattern = r'<user_intention>(.*?)</user_intention>'
        user_intention_match = re.search(user_intention_pattern, intention_response, re.DOTALL)
        if user_intention_match:
            user_intention = user_intention_match.group(1)

        inquiry_type = ''
        inquiry_type_pattern = r'<inquiry_type>(.*?)</inquiry_type>'
        inquiry_type_match = re.search(inquiry_type_pattern, intention_response, re.DOTALL)
        if inquiry_type_match:
            inquiry_type = inquiry_type_match.group(1)

        assistant_response['intention_thought'] = user_intention.replace('<summary>', '') + '\n' + inquiry_type.replace('<summary>', '')

        # 2. Intention summarization
        if '<summary>' in intention_response:
            summary_content = f"用户意图：\n{user_intention}```根据以上信息，完成你的任务，总结用户意图。"
            messages = get_simple_messages(assistant_prompt_summary, summary_content)
            summary_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
            count_tokens(current_usage, step_usage)
            # print(summary_response)
            assistant_response['summary'] = summary_response
            self.update_history_details(user_response, assistant_response)
            return assistant_response, user_intention, current_usage

        # 3. Emotion
        emotion_content = f"```本轮对话：\n{current_history}```根据以上信息，完成你的任务，输出你的判断。"
        messages = get_simple_messages(assistant_prompt_emotion, emotion_content)
        emotion_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(emotion_response)
        assistant_response['emotion_thought'] = emotion_response.replace('<anxiety>', '')

        # Trigger 3. Intention summarization
        if '<anxiety>' in emotion_response:
            summary_content = f"用户意图：\n{user_intention}```根据以上信息，完成你的任务，总结用户意图。"
            messages = get_simple_messages(assistant_prompt_summary, summary_content)
            summary_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
            count_tokens(current_usage, step_usage)
            # print(summary_response)
            assistant_response['summary'] = summary_response
            self.update_history_details(user_response, assistant_response)
            return assistant_response, user_intention, current_usage

        # 4. Reference options
        options_content = f"```用户意图：\n{user_intention}```\n```{inquiry_type}```\n根据以上信息，完成你的任务。"
        messages = get_simple_messages(assistant_prompt_options, options_content)
        options_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(options_response)
        options_pattern = r'<ref_option>(.*?)</ref_option>'
        options_match = re.search(options_pattern, options_response, re.DOTALL)
        if options_match:
            options = options_match.group(1)
            assistant_response['options_thought'] = options_response[:options_response.find('<')] + '参考选项：' + options
        else:
            options = options_response
            assistant_response['options_thought'] = options_response

        # 5. Respond to user
        response_content = f"```本轮对话：\n{current_history}```\n```用户意图：\n{user_intention}```\n```{inquiry_type}```\n```参考选项：\n{options}```\n根据以上信息和你的任务，回复用户。"
        messages = get_simple_messages(assistant_prompt_response, response_content)
        response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(response)
        assistant_response['content'] = response

        # Update dialogue history
        self.update_history(None, response)
        self.update_history_details(user_response, assistant_response)
        return assistant_response, user_intention, current_usage
