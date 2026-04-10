#!/usr/bin/env python3
"""
Preprocess Hu Chenfeng (户晨风) livestream transcripts into a ShareGPT-style
conversations dataset for SFT fine-tuning Gemma with Unsloth.

INPUT
-----
HuChenFeng/<YYYY年MM月>/<YYYY-MM-DD*.md>
Each line is one turn, prefixed with a full-width-colon speaker label:
    户晨风：....   (host  -> assistant, the persona we want to mimic)
    某网友：....   (viewer -> user)
Turns are separated by blank lines.

OUTPUT
------
A JSONL file. Each line is one training example:

    {
      "conversations": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
      ],
      "source":      "2024年06月/2024-06-09.md",
      "chunk_index": 3
    }

This is exactly the shape Unsloth's `standardize_data_formats` expects, and is
a drop-in replacement for `mlabonne/FineTome-100k` in the Gemma SFT notebook.

CLEANING
--------
- Donor/gift thanks like "感谢海马斯" / "谢谢印象海绵八万克" are stripped from
  inside turns (random user IDs that would otherwise pollute the model).
  Legitimate "感谢大家 / 谢谢主播 / 感谢支持" etc. are kept.
- Pure-filler user turns ("嗯。", "啊？", "对。") are dropped — they're zero
  signal and usually different viewers' independent reactions.
- Repeated punctuation (……………… / ，，， / ！！！) is collapsed.
- Filler words and口头禅 (是吧 / 你听我讲 / 我我我 / 刚开播刚开播) are KEPT
  on purpose — they are the host's style and the whole point of the finetune.

MODES
-----
--mode pair  (default, recommended)
    One example per single (某网友, 户晨风) pair. Most honest framing for a
    livestream where adjacent viewer comments are different people.

--mode window
    Multi-turn windows up to --turns-per-chunk turns. Short user turns
    (<= --boundary-user-chars chars) act as topic boundaries to avoid gluing
    unrelated viewer comments together. A system prompt is prepended to the
    first user turn explaining the livestream context.

USAGE
-----
    python prepare_dataset.py --mode pair   --output hcf_sft_pair.jsonl
    python prepare_dataset.py --mode window --output hcf_sft_window.jsonl

Then in the training notebook (mirrors the unsloth tutorial verbatim):

    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template, standardize_data_formats

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    dataset = load_dataset("json", data_files="hcf_sft_pair.jsonl", split="train")
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for c in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

HOST_LABEL = "户晨风"  # the persona we are training the model to mimic
VIEWER_LABEL = "某网友"  # everything a chat viewer says

ROLE_MAP = {
    HOST_LABEL: "assistant",
    VIEWER_LABEL: "user",
}

# Lines look like "户晨风：blah blah" using the full-width colon U+FF1A.
TURN_RE = re.compile(rf"^({HOST_LABEL}|{VIEWER_LABEL})：(.*)$")

# Only process daily transcripts: 2023-03-10.md, 2023-03-12-INC.md, etc.
DATE_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}.*\.md$")


# ============================================================
# Cleaning
# ============================================================

# Donor / gift thanks: "感谢海马斯", "谢谢印象海绵八万克", "感谢Razer".
# We must NOT strip legitimate "感谢大家 / 谢谢主播 / 感谢支持" — those are
# protected by the allowlist below.
_THANKS_RE = re.compile(r"(?:感谢|谢谢)([\u4e00-\u9fa5A-Za-z0-9_]{2,15})")

_THANKS_KEEP_PREFIXES = (
    "大家",
    "你们",
    "您们",
    "主播",
    "老师",
    "父母",
    "家人",
    "兄弟",
    "观众",
    "朋友",
    "支持",
    "理解",
    "配合",
    "关注",
    "陪伴",
    "捧场",
    "鼓励",
    "提醒",
    "光临",
    "信任",
    "上麦",
    "连麦",
)

_REPEAT_DOT = re.compile(r"…{3,}")
_REPEAT_COMMA = re.compile(r"[，,]{2,}")
_REPEAT_BANG = re.compile(r"[！!]{2,}")
_REPEAT_QUEST = re.compile(r"[？?]{2,}")
_LEADING_PUNCT = re.compile(r"^[，。、,. ]+")
_INNER_SPACES = re.compile(r" {2,}")

# Pure-filler USER turns to drop entirely. Host turns are NEVER dropped on
# this basis — when 户晨风 says just "啊？" it is part of his cadence.
_FILLER_CHARS = set(
    "嗯啊哦哎呀呃唉哈嘿哼哇嗨好对是的不行可以来噢诶咦呵哦哦 ，。！？,.!?"
)


def _strip_thanks(match: re.Match) -> str:
    tail = match.group(1)
    if any(tail.startswith(p) for p in _THANKS_KEEP_PREFIXES):
        return match.group(0)
    return ""


def clean_text(s: str) -> str:
    s = _THANKS_RE.sub(_strip_thanks, s)
    s = _REPEAT_DOT.sub("……", s)
    s = _REPEAT_COMMA.sub("，", s)
    s = _REPEAT_BANG.sub("！", s)
    s = _REPEAT_QUEST.sub("？", s)
    s = _LEADING_PUNCT.sub("", s)
    s = _INNER_SPACES.sub(" ", s)
    return s.strip()


def is_filler_user_turn(s: str) -> bool:
    """True for tiny user turns that are entirely interjections / particles."""
    return 0 < len(s) <= 4 and all(c in _FILLER_CHARS for c in s)


# ============================================================
# Parsing
# ============================================================


def parse_turns(md_path: Path) -> list[dict]:
    """Parse one transcript file into a flat alternating list of turns,
    applying cleaning and filler-drop. Adjacent same-role turns are merged
    so the result strictly alternates user / assistant — Gemma's chat
    template requires that.
    """
    raw: list[dict] = []
    with md_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = TURN_RE.match(line)
            if not m:
                continue
            label, body = m.group(1), m.group(2).strip()
            if not body:
                continue
            body = clean_text(body)
            if not body:
                continue
            role = ROLE_MAP[label]
            if role == "user" and is_filler_user_turn(body):
                continue
            raw.append({"role": role, "content": body})

    merged: list[dict] = []
    for t in raw:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] = merged[-1]["content"] + " " + t["content"]
        else:
            merged.append(dict(t))
    return merged


# ============================================================
# Mode 1: single (user, assistant) pairs
# ============================================================


def emit_pairs(
    turns: list[dict],
    *,
    min_chars: int,
    max_chars_per_turn: int,
) -> Iterator[list[dict]]:
    """One training example per (user, assistant) pair.

    The most honest framing for a livestream Q&A: each viewer comment is
    independent, gets one host response, and there is no false multi-turn
    coherence between unrelated comments.
    """
    for i in range(len(turns) - 1):
        if turns[i]["role"] != "user" or turns[i + 1]["role"] != "assistant":
            continue
        u = turns[i]["content"]
        a = turns[i + 1]["content"]
        if len(u) > max_chars_per_turn or len(a) > max_chars_per_turn:
            continue
        if len(u) + len(a) < min_chars:
            continue
        yield [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]


# ============================================================
# Mode 2: livestream-context windows
# ============================================================

DEFAULT_WINDOW_SYSTEM_PROMPT = (
    "你是户晨风，正在直播。下面 user 角色的发言来自直播间不同观众的弹幕，"
    "相邻弹幕未必出自同一个人，也未必相互关联。请按你一贯的风格回应。"
)


def emit_windows(
    turns: list[dict],
    *,
    turns_per_chunk: int,
    min_turns: int,
    min_chars: int,
    max_chars_per_turn: int,
    boundary_user_chars: int,
    system_prompt: str,
) -> Iterator[list[dict]]:
    """Multi-turn windows.

    Short user turns (<= boundary_user_chars chars) act as topic boundaries:
    they are likely independent弹幕 from a different viewer, so we cut the
    window there and skip them rather than glue them into the conversation.
    """
    cleaned: list[dict] = []
    for t in turns:
        if len(t["content"]) > max_chars_per_turn:
            continue
        if cleaned and cleaned[-1]["role"] == t["role"]:
            cleaned[-1]["content"] = cleaned[-1]["content"] + " " + t["content"]
        else:
            cleaned.append(dict(t))

    n = len(cleaned)
    i = 0
    while i < n and cleaned[i]["role"] != "user":
        i += 1

    while i < n:
        end = min(i + turns_per_chunk, n)
        # Topic boundary: cut at the next very-short user turn after i.
        for j in range(i + 1, end):
            if (
                cleaned[j]["role"] == "user"
                and len(cleaned[j]["content"]) <= boundary_user_chars
            ):
                end = j
                break
        # Trim back so the chunk ends on an assistant turn.
        while end > i and cleaned[end - 1]["role"] != "assistant":
            end -= 1
        chunk = [dict(t) for t in cleaned[i:end]]
        if (
            len(chunk) >= min_turns
            and chunk[0]["role"] == "user"
            and chunk[-1]["role"] == "assistant"
            and sum(len(t["content"]) for t in chunk) >= min_chars
        ):
            if system_prompt:
                chunk[0]["content"] = f"{system_prompt}\n\n{chunk[0]['content']}"
            yield chunk

        # Advance past this window.
        i = max(end, i + 1)
        # Skip the boundary turn itself if we stopped on one.
        if (
            i < n
            and cleaned[i]["role"] == "user"
            and len(cleaned[i]["content"]) <= boundary_user_chars
        ):
            i += 1
        while i < n and cleaned[i]["role"] != "user":
            i += 1


# ============================================================
# Driver
# ============================================================


def iter_transcripts(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.md")):
        if DATE_FILE_RE.match(p.name):
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("HuChenFeng"),
        help="Root directory containing YYYY年MM月/YYYY-MM-DD*.md files",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("hcf_sft.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument(
        "--mode",
        choices=["pair", "window"],
        default="pair",
        help="pair: one example per (user,assistant) pair (default, most honest); "
        "window: multi-turn livestream windows with system framing",
    )
    ap.add_argument(
        "--turns-per-chunk",
        type=int,
        default=8,
        help="[window mode] max turns per training example",
    )
    ap.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="[window mode] drop windows with fewer turns than this",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Drop examples whose total content is shorter than this many chars",
    )
    ap.add_argument(
        "--max-chars-per-turn",
        type=int,
        default=1500,
        help="Drop turns longer than this many characters (likely monologues)",
    )
    ap.add_argument(
        "--boundary-user-chars",
        type=int,
        default=6,
        help="[window mode] user turns this short or shorter act as topic boundaries",
    )
    ap.add_argument(
        "--system-prompt",
        default=DEFAULT_WINDOW_SYSTEM_PROMPT,
        help="[window mode] framing prepended to the first user turn; "
        "pass empty string to disable",
    )
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        ap.error(f"Input dir does not exist: {input_dir}")

    n_files = n_clean_turns = n_examples = total_chars = 0
    with args.output.open("w", encoding="utf-8") as out:
        for md_path in iter_transcripts(input_dir):
            n_files += 1
            turns = parse_turns(md_path)
            n_clean_turns += len(turns)
            rel = md_path.relative_to(input_dir).as_posix()

            if args.mode == "pair":
                gen = emit_pairs(
                    turns,
                    min_chars=args.min_chars,
                    max_chars_per_turn=args.max_chars_per_turn,
                )
            else:
                gen = emit_windows(
                    turns,
                    turns_per_chunk=args.turns_per_chunk,
                    min_turns=args.min_turns,
                    min_chars=args.min_chars,
                    max_chars_per_turn=args.max_chars_per_turn,
                    boundary_user_chars=args.boundary_user_chars,
                    system_prompt=args.system_prompt,
                )

            for idx, chunk in enumerate(gen):
                record = {
                    "conversations": chunk,
                    "source": rel,
                    "chunk_index": idx,
                }
                out.write(json.dumps(record, ensure_ascii=False))
                out.write("\n")
                n_examples += 1
                total_chars += sum(len(t["content"]) for t in chunk)

    print(f"Mode:              {args.mode}")
    print(f"Files processed:   {n_files}")
    print(f"Cleaned turns:     {n_clean_turns}")
    print(f"Examples written:  {n_examples}")
    print(f"Total content:     {total_chars / 1e6:.1f} M chars")
    print(f"Output:            {args.output}")


if __name__ == "__main__":
    main()
