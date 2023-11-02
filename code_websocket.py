import argparse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from example_completion_websocket import CodeSocket

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

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

code = CodeSocket(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature, top_p=top_p,
                  max_seq_len=max_seq_len, max_batch_size=max_batch_size, max_gen_len=max_gen_len)
generator = code.get_generator()


@app.post("/generate_code")
async def generate_code(request: Request):
    msgs = request.messages
    prompts = msgs.prompts

    # Initialize and configure the CodeSocket

    result = code.get_result(generator=generator, prompts=prompts)

    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = 19324
    uvicorn.run(app, host=host, port=port)
