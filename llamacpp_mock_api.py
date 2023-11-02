from typing import Optional
import asyncio
import fire
from flask import Flask, jsonify, request
import torch.distributed as dist
from gevent import pywsgi
import uvicorn
from llama import Llama
import logging
from uvicorn import Server
logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    port: int = 19324,
):
    # Create our Code Llama object.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # With torchrun and distributed PyTorch, multiple copies of this code
    # can be run at once. We only want one of them (node 0) to have the Flask API
    # and we will use it to control the rest.
    if dist.get_rank() == 0:
        app = Flask(__name__)
        
        def run_chat_completion(prompts):
            # Broadcast what should be processed to other nodes (acting as a C&C node).
            # dist.broadcast_object_list([instructions, max_gen_len, temperature, top_p])
            # print(instructions)
            # Start Code Llama inferencing.
            print(prompts)
            results =generator.text_completion(
                prompts=prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            print(results)

            # Send the response back.

            return results

        @app.route("/v1/completions", methods=["POST"])
        def completions():
            content = request.json
            
            # Is used by Continue to generate a relevant title corresponding to the
            # model's response, however, the current prompt passed by Continue is not
            # good at obtaining a title from Code Llama's completion feature so we
            # use chat completion instead.
            messages = [
                {
                    "role": "user",
                    "content": content["prompt"]
                }
            ]
            
            # Perform Code Llama chat completion.
            response = run_chat_completion([messages])
            
            # Send back the response.
            return jsonify({"choices": [{"text": response}]})

        @app.route("/v1/chat/completions", methods=["POST"])
        async def chat_completions():
            content = request.json
            messages = content["messages"]
            print(messages)
            # print(messages)
            # # Continue does not follow the user-assistant turn constraints Code Llama
            # # needs. It has duplicate subsequent responses for a role. For example, a/u/u/a
            # # will be sent by Continue when Code Llama only supports u/a/u/a so we squash
            # # duplicate subsequent roles into a single message.
            # if messages[0]["role"] == "assistant":
            #     messages[0]["role"] = "system"
            # last_role = None
            # remove_elements = []
            # for i in range(len(messages)):
            #     if messages[i]["role"] == last_role:
            #         messages[i-1]["content"] += "\n\n" + messages[i]["content"]
            #         remove_elements.append(i)
            #     else:
            #         last_role = messages[i]["role"]
            # for element in remove_elements:
            #     messages.pop(element)
            print('aaaaaaaaa')
            # Perform Code Llama chat completion.
            prompts = messages[0]["prompts"]
            response = await run_chat_completion([prompts])
            result = await response
            print('bbbbbbbbbbbb')
            # Send JSON with Code Llama's response back to the VSCode Continue
            # extension. Note the extension expects six characters preappended to the
            # reponse JSON so we preappend the random string "onesix" to fulfill that requirement.
            print(result)
            for prompt, result in zip(prompts, response):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
            return {"msg": 'success'}

        # Run the Flask API server.
        # app.run(host="0.0.0.0",port=port)
            # 在此处运行 Flask 应用，使用 uvicorn 服务器



        server = Server(app, host="0.0.0.0", port=19324, log_level="info")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server.serve())

    
    # Nodes which are not node 0 wait for tasks.
    else:
        while True:
            config = [None] * 4
            try:
                dist.broadcast_object_list(config)
                generator.chat_completion(
                    config[0], max_gen_len=config[1], temperature=config[2], top_p=config[3]
                )
            except:
                pass

if __name__ == "__main__":
    fire.Fire(main)