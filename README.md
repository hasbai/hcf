---
language:
  - zh
license: gemma
base_model: google/gemma-4-E4B-it
tags:
  - gemma
  - gemma-4
  - chinese
  - lora
  - gguf
  - unsloth
  - sft
  - conversational
pipeline_tag: text-generation
---

# hcf-gemma-4

基于 [Gemma 4 E4B Instruct](https://huggingface.co/google/gemma-4-E4B-it) 微调的中文对话模型，训练目标是模仿知名直播主播**户晨风**的说话风格与语气。

---

## 模型简介

| 项目 | 详情 |
|------|------|
| 基础模型 | `google/gemma-4-E4B-it`（通过 unsloth 加速） |
| 微调方法 | LoRA（r=8, lora_alpha=8，仅更新语言层） |
| 训练数据 | 户晨风 2023–2025 年直播文字稿，489 个文件 |
| 训练样本 | 167,275 条单轮问答对 |
| 训练步数 | 2,000 步（Tesla T4，4-bit 量化） |
| 可训练参数 | 18,350,080 / 8,014,506,528（约 0.23%） |
| 发布格式 | GGUF Q8_0 |

---

## 数据集与预处理

训练数据来源于 [HuChenFeng](https://github.com/Olcmyk/HuChenFeng) 项目整理的直播文字稿，格式为每行一句、全角冒号分隔说话人：

```
某网友：你觉得这个事情怎么看？
户晨风：哎，我跟你讲，这个事情吧……
```

预处理脚本（[`prepare_dataset.py`](./prepare_dataset.py)）的主要步骤：

- **说话人映射**：`某网友 → user`，`户晨风 → assistant`
- **礼物感谢清洗**：去除 `感谢XXX / 谢谢XXX`（随机用户 ID），保留 `感谢大家 / 谢谢主播` 等通用表达
- **冗余标点折叠**：`…………` → `……`，`！！！` → `！` 等
- **零信号用户回合过滤**：纯语气词（如"嗯""哦"，≤4字）的用户回合直接丢弃
- **相邻同角色合并**：保证严格的 user/assistant 交替，符合 Gemma-4 chat template 要求

微调保留了户晨风的口头禅（"是吧""你听我讲""刚开播刚开播"等），这正是风格迁移的目标。

---

## 使用方式

### 使用 llama.cpp / Ollama（推荐）

```bash
# 下载 GGUF 文件后
llama-cli -m hcf-gemma-4-Q8_0.gguf \
  --chat-template gemma \
  -p "电车买什么好？" \
  --temp 1.0 --top-p 0.95 --top-k 64
```

### 使用 Python（transformers + unsloth）

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastModel.from_pretrained(
    model_name="hasbai/hcf-gemma-4",
    max_seq_length=1024,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

messages = [{"role": "user", "content": "你觉得年轻人现在最应该做什么？"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=1.0, top_p=0.95, top_k=64,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **推荐推理参数**：`temperature=1.0, top_p=0.95, top_k=64`（与 Gemma-4 官方推荐一致）

---

## 局限性与声明

- **风格迁移，非知识增强**：本模型的目标是学习户晨风的语言风格，不代表其观点正确性，也不构成任何形式的建议。
- **直播语料特点**：训练数据为口语化直播文字稿，模型输出会带有较强的口语色彩、语气词与重复表达，这是有意为之。
- **基础模型限制**：本模型继承 Gemma-4 E4B 的能力边界，复杂推理、专业知识等方面仍受基础模型限制。
- **版权说明**：训练数据文字稿由第三方整理，本项目仅用于学术与技术研究目的。

---

## 训练复现

```bash
# 1. 克隆仓库（含语料子模块）
git clone --recurse-submodules https://github.com/hasbai/hcf
cd hcf

# 2. 生成训练数据
python prepare_dataset.py --mode pair --output hcf_sft_pair.jsonl

# 3. 在 Colab 运行训练 notebook
#    见 Gemma4_(E4B)-Text.ipynb
```

---

## 相关链接

- 训练脚本与预处理代码：[hasbai/hcf](https://github.com/hasbai/hcf)
- 原始文字稿语料：[Olcmyk/HuChenFeng](https://github.com/Olcmyk/HuChenFeng)
- 基础模型：[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- 训练框架：[unsloth](https://github.com/unslothai/unsloth)
