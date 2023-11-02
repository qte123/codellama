# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
os.environ["PYTHONIOENCODING"] = "utf-8"
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import  numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class InfillingPrediction(TypedDict, total=False):
    generation: str
    full_text: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
            ckpt_dir: str,
            tokenizer_path: str,
            max_seq_len: int,
            max_batch_size: int,
            model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            if device == "cuda":
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("gloo")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device == "cuda":
            torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        # support for mac
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

   # 计算TPS
    def get_token_per_second(self,chunks, elapsed_time):
        # 计算输出token数量
        output_token_count = len(chunks)
        # 计算TPS
        token_per_second = output_token_count / elapsed_time
        return token_per_second

   # 计算WPS
    def get_word_per_second(self,chunks,elapsed_time):
        # 计算总单词长度
        total_words_len = sum(len(chunk) for chunk in chunks)
        # 计算单词每秒数
        word_per_second=total_words_len/elapsed_time
        return word_per_second

    # 数据处理和曲线绘制
    def create_plot(self,outputs, max_token, name,is_gpu=True):
        # 指定 data 文件夹的路径
        data_folder_plt = "data/plt"
        # 生成带有时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"output_{timestamp}"
        outputs = pd.DataFrame(outputs)
        # 计算除第一行外的数据行数量
        list_length = len(outputs)

        # 根据数据行数量选择 token_length 的长度
        token_length = outputs["token_length"][0 : list_length + 1]

        # 提取数据列
        spend_times = outputs["spend_time"][0 : list_length + 1]
        token_per_second = outputs["token_per_second"][0 : list_length + 1]
        word_per_second = outputs["word_per_second"][0 : list_length + 1]

        # 获取数据中的最小值和最大值
        min_token_length = np.min(token_length)
        max_token_length = np.max(token_length)

        # 设置x轴的范围
        x_length = [min_token_length, max_token]
        # 设置y轴的范围
        y_time_length = [0,150]
        y_tps_length=[0,30]
        y_wps_length=[0,50]
        #表格的行高
        row_height = 0.3
        # 创建图像
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # 指定需要标注的 x 坐标
        x_ticks = np.arange(0, max_token + 1, 100)
        #指定需要标注的x的坐标
        x_ticks_ = np.arange(0, max_token + 1, 200)
        # 使用 numpy 的 isin 函数筛选出满足条件的数据点下标
        indices = np.isin(token_length, x_ticks)
        selected_x = token_length[indices]
        indices_ = np.isin(token_length, x_ticks_)
        selected_x_ = token_length[indices_]
        # 绘制 "spend_time" 的曲线图
        ax1.plot(token_length, spend_times, marker="", linestyle="-", color="b")
        # 获取满足条件的 x 坐标及对应的 y 坐标
        selected_y_time = spend_times[indices]
        selected_y_time_ = spend_times[indices_]
        #设置表格
        x_tables = np.append(selected_x, [token_length.iloc[-1]])
        y_tables_time = np.append(selected_y_time, [spend_times.iloc[-1]])
        data1 = {
        "Token Length": x_tables,
        "Spend Time": y_tables_time
        }
        df1 = pd.DataFrame(data1)
        for x, y in zip(selected_x, selected_y_time):
            ax1.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
            ax1.scatter(x, y, color="black")  # 添加散点图，用红色表示
        # 标注曲线的最后一个点
        ax1.annotate(
            f"{spend_times.iloc[-1]:.2f}",
            (token_length.iloc[-1], spend_times.iloc[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        
        # # 显示表格数据 1
        # cell_text1 = []
        # for row in range(len(df1)):
        #     rounded_values = df1.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        #     cell_text1.append(rounded_values.values)
        # table1=ax1.table(cellText=cell_text1, colLabels=df1.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])
        
        # table1.set_row_heights([row_height] * len(data1))
        ax1.scatter(token_length.iloc[-1], spend_times.iloc[-1], color="black")
        ax1.set_xlabel("Token Length")
        ax1.set_ylabel("Spend Time")
        ax1.set_title("Spend Time by Token Length")
        ax1.grid(True)
        # 设置 x 轴范围
        ax1.set_xlim(x_length)
        # 设置 x 轴刻度的间隔
        ax1.set_xticks(x_ticks)
        ax1.set_ylim(y_time_length)

        # 绘制 "token_efficiency" 的曲线图
        ax2.plot(token_length, token_per_second, marker="", linestyle="-", color="r")
        selected_y_efficiencies = token_per_second[indices]
        selected_y_efficiencies_ = token_per_second[indices_]
        y_tables_efficiencies=np.append(selected_y_efficiencies,[token_per_second.iloc[-1]])
        data2 = {
        "Token Length": x_tables,
        "Token Efficiency": y_tables_efficiencies
        }
        df2=pd.DataFrame(data2)
        for x, y in zip(selected_x, selected_y_efficiencies):
            ax2.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
            ax2.scatter(x, y, color="black")  # 添加散点图，用红色表示
        ax2.annotate(
        f"{token_per_second.iloc[-1]:.2f}",
        (token_length.iloc[-1], token_per_second.iloc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        )
        
        # # 显示表格数据 2
        # cell_text2 = []
        # for row in range(len(df2)):
        #     rounded_values = df2.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        #     cell_text2.append(rounded_values.values)
        # table2=ax2.table(cellText=cell_text2, colLabels=df2.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])
        # table2.set_row_heights([row_height] * len(data2))
        ax2.scatter(token_length.iloc[-1], token_per_second.iloc[-1], color="black")
        ax2.set_xlabel("Token Length")
        ax2.set_ylabel("Token per second (TPS)")
        ax2.set_title("Token per second (TPS) by Token Length")
        ax2.grid(True)
        # 设置 x 轴范围
        ax2.set_xlim(x_length)
        # 设置 x 轴刻度的间隔
        ax2.set_xticks(x_ticks)
        ax2.set_ylim(y_tps_length)

        # 绘制 "token_aver_efficiency" 的曲线图
        ax3.plot(token_length, word_per_second, marker="", linestyle="-", color="g")
        selected_y_aver_efficiencies = word_per_second[indices]
        selected_y_aver_efficiencies_ = word_per_second[indices_]
        #设置表格
        y_tables_aver_efficiencies=np.append(selected_y_aver_efficiencies,[word_per_second.iloc[-1]])
        data3 = {
        "Token Length": x_tables,
        "Token Average Efficiency": y_tables_aver_efficiencies
        }
        df3 = pd.DataFrame(data3)
        for x, y in zip(selected_x, selected_y_aver_efficiencies):
            ax3.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
            ax3.scatter(x, y, color="black")  # 添加散点图，用红色表示
        ax3.annotate(
        f"{word_per_second.iloc[-1]:.2f}",
        (token_length.iloc[-1], word_per_second.iloc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        )

        # # 显示表格数据 3
        # cell_text3 = []
        # for row in range(len(df3)):
        #     rounded_values = df3.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
        #     cell_text3.append(rounded_values.values)
        # table3=ax3.table(cellText=cell_text3, colLabels=df3.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])
        # 设置单元格高度
        # table3.set_row_heights([row_height] * len(data3))
        ax3.scatter(token_length.iloc[-1], word_per_second.iloc[-1], color="black")
        ax3.set_xlabel("Token Length")
        ax3.set_ylabel("Word per second (WPS)")
        ax3.set_title("Word per second (WPS) by Token Length")
        ax3.grid(True)
        # 设置 x 轴范围
        ax3.set_xlim(x_length)
        # 设置 x 轴刻度的间隔
        ax3.set_xticks(x_ticks)
        ax3.set_ylim(y_wps_length)
        

        plt.subplots_adjust(wspace=0.4)
        # 设置总标题
        if is_gpu:
            num_gpus = torch.cuda.device_count()
            fig.suptitle(
            f"{name}(GPUS{num_gpus}) Generation Efficiency",
            fontsize=16,
            fontweight="bold",
        )
        else:
            fig.suptitle(
                f"{name}(ONLY CPU) Generation Efficiency",
                fontsize=16,
                fontweight="bold",
            )

        # 生成带有时间戳的文件名
        png_filename = f"{name}_{filename}_gpus{num_gpus}.png"
        png_filepath = os.path.join(data_folder_plt, png_filename)
        # plt.tight_layout()
        # 保存 PNG 文件
        plt.savefig(png_filepath)
        # 关闭图形
        plt.close(fig)
        # 显示图像
        # plt.show()


    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[List[int]],
            max_gen_len: int,
            temperature: float = 0.6,
            top_p: float = 0.9,
            logprobs: bool = False,
            echo: bool = False,
            stop_token: Optional[int] = None,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        average_start_time = time.time()
        if stop_token is None:
            stop_token = self.tokenizer.eos_id
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float, device=device)

        prev_pos = 0
        chunks = []
        outputs = []
        stop_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1: cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1: cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            # print(next_token.tolist())
            chunk = self.tokenizer.decode(next_token.tolist())
            end_time = time.time()  # 获取结束时间戳
            chunks.append(chunk)
            token_length = len(chunks)
            spend_time = end_time - average_start_time  # 计算平均输出时间
            token_per_second = self.get_token_per_second(chunks, spend_time)  # 计算平均token输出效率
            word_per_second = self.get_word_per_second(chunks,spend_time)
            output = {
                "token_length": token_length,
                "spend_time": spend_time,
                "token_per_second": token_per_second,
                "word_per_second": word_per_second
            }
            print(output['token_length'])
            outputs.append(output)
            # print(outputs)
            # print(chunk)
            # print("".join(chunks).encode("utf-8").decode("utf-8"))
            # 计算新生成的chunk
            # print(tokens.tolist())


            stop_reached |= (~input_text_mask[:, cur_pos]) & (next_token == stop_token)
            prev_pos = cur_pos
            if all(stop_reached):
                break
        self.create_plot(outputs,1000,'code-exllama-13B',is_gpu=True)
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []

        for i, toks in enumerate(tokens.tolist()):
            # print(toks)
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start: len(prompt_tokens[i]) + max_gen_len]
            # cut to stop token if present
            if stop_token in toks:
                stop_idx = toks.index(stop_token)
                toks = toks[:stop_idx]
                probs = probs[:stop_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
            self,
            prompts: List[str],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
            echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def text_infilling(
            self,
            prefixes: List[str],
            suffixes: List[str],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
            suffix_first: bool = False,
    ) -> List[InfillingPrediction]:
        assert self.tokenizer.eot_id is not None
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [
            infilling_prompt_tokens(
                self.tokenizer, prefix, suffix, suffix_first=suffix_first
            )
            for prefix, suffix in zip(prefixes, suffixes)
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=False,
            stop_token=self.tokenizer.eot_id,
        )

        generations = [self.tokenizer.decode_infilling(t) for t in generation_tokens]

        if logprobs:
            return [
                {
                    "generation": generation,
                    "logprobs": logprobs_i,
                    "tokens": t,
                    "full_text": prefix + generation + suffix,
                }
                for prefix, suffix, generation, t, logprobs_i in zip(
                    prefixes,
                    suffixes,
                    generations,
                    generation_tokens,
                    generation_logprobs,
                )
            ]
        else:
            return [
                {
                    "generation": generation,
                    "full_text": prefix + generation + suffix,
                }
                for prefix, suffix, generation in zip(prefixes, suffixes, generations)
            ]

    def chat_completion(
            self,
            dialogs: List[Dialog],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                             {
                                 "role": dialog[1]["role"],
                                 "content": B_SYS
                                            + dialog[0]["content"]
                                            + E_SYS
                                            + dialog[1]["content"],
                             }
                         ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
                ],
                [],
            )
            assert (
                    dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def infilling_prompt_tokens(
        tokenizer: Tokenizer,
        pre: str,
        suf: str,
        suffix_first: bool = False,
) -> List[int]:
    """
    Format and encode an infilling problem.
    If `suffix_first` is set, format in suffix-prefix-middle format.
    """
    assert tokenizer.prefix_id is not None
    assert tokenizer.middle_id is not None
    assert tokenizer.suffix_id is not None
    if suffix_first:
        # format as "<PRE> <SUF>{suf} <MID> {pre}"
        return (
                [tokenizer.bos_id, tokenizer.prefix_id, tokenizer.suffix_id]
                + tokenizer.encode_infilling(suf)
                + [tokenizer.middle_id]
                + tokenizer.encode(pre, bos=False, eos=False)
        )
    else:
        # format as "<PRE> {pre} <SUF>{suf} <MID>"
        return (
                [tokenizer.bos_id, tokenizer.prefix_id]
                + tokenizer.encode(pre, bos=False, eos=False)
                + [tokenizer.suffix_id]
                + tokenizer.encode_infilling(suf)
                + [tokenizer.middle_id]
        )
