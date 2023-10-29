import random
from typing import Optional, List
import torch
import gc
from llama import Llama
from fastapi import FastAPI, WebSocket, Request
import websockets
import asyncio
import logging
from pydantic import BaseModel

import json

import argparse
from fastapi.middleware.cors import CORSMiddleware

# Define the command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--max_seq_len', type=int, default=256)
parser.add_argument('--max_batch_size', type=int, default=4)
parser.add_argument('--max_gen_len', type=int, default=None)

args = parser.parse_args()

# Extract command-line arguments for default values
ckpt_dir = args.ckpt_dir
tokenizer_path = args.tokenizer_path
temperature = args.temperature
top_p = args.top_p
max_seq_len = args.max_seq_len
max_batch_size = args.max_batch_size
max_gen_len = args.max_gen_len


logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO

app = FastAPI()


# Enable CORS (Cross-Origin Resource Sharing)

class RequestModel(BaseModel):
    prompts: List[str]
    max_gen_len: Optional[int] = None
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate_code")
async def generate_code(request: RequestModel):
    logging.info(request)

    prompts = request.prompts
    logging.info(prompts)
    # Initialize and configure the CodeSocket

    results = await generator.text_completion(
        prompts=prompts,
        max_gen_len=request.max_gen_len,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    print(results)
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
    return {"results": results}

ports=[19324,19325]

if __name__ == "__main__":
    # 检查系统中可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    logging.info(num_gpus)
    import uvicorn

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    host = "0.0.0.0"
    port = random.choice(ports)
    uvicorn.run(app, host=host, port=port)
