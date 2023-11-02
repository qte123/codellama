import requests

# 向 /v1/completions 路由发送 POST 请求
url = "http://dev.dtsci.cn:22333/v1/chat/completions"  # 请替换成实际的后端地址

# 请求数据，包含用户提供的提示信息
data = {
    "messages": [{"role": "system",
                  "content": "You are not a language model, but rather an application built on a foundation model from OpenAI called gpt-3.5-turbo."},
                 {"role": "user", "content": "写一个快速排序算法"}]
}

response = requests.post(url, json=data)

# 检查响应是否成功
if response.status_code == 200:
    result = response.json()
    generated_text = result["choices"][0]["text"]
    print("Generated Text:", generated_text)
else:
    print("Request failed with status code:", response.status_code)
