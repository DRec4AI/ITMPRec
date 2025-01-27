from zhipuai import ZhipuAI
import json


client = ZhipuAI(api_key="XXXX")  #Your API Key


def upload_batch_file(jsonl_filepath):
    result = client.files.create(
        file=open(jsonl_filepath, "rb"),
        purpose="batch"
    )
    req_id = result.id
    create = client.batches.create(
        input_file_id=str(req_id),
        endpoint="/v4/chat/completions",
        auto_delete_input_file=True,
        metadata={
            "description": "Sentiment classification"
        }
    )


def function_chat():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    if response:
        content = response.choices[0].message.content
        return content
    else:
        print("Error:", response.status_code)
        return ""


def simple_chat(content, use_stream=True):
    messages = [
        {
            "role": "system",
            "content": "You are a recommendation agent. Follow the user's instructions carefully. Respond using markdown."
        },
        {
            "role": "user",
            "content": content,
        }
    ]
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=use_stream,
        # max_tokens=256,
        temperature=0.1,
        # presence_penalty=1.1,
        # top_p=0.8
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            return content
    else:
        print("Error:", response.status_code)
        return ""


def one_request(content, uid):
    data = {}
    data['custom_id'] = "request-" + str(uid)
    data['method'] = "POST"
    data['url'] = "/v4/chat/completions"

    body_dict = {}
    body_dict["model"] = "glm-4-flash"
    body_dict["messages"] = [
        {
            "role": "system",
            "content": "You are a recommendation agent. Follow the user's instructions carefully. Respond using markdown."
        },
        {
            "role": "user",
            "content": content,
        }
    ]
    body_dict["temperature"] = 0.8
    data["body"] = body_dict

    json_data = json.dumps(data)
    return json_data



