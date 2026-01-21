#!/usr/bin/env python3
"""
ytsum.py — YouTube URL → Audio → Whisper Transcript → AI Notes → Markdown

Pipeline:
1) Download audio from YouTube (yt-dlp)
2) Transcribe with Whisper (local)
3) Summarize chunk-by-chunk with an LLM
4) Synthesize into polished Markdown notes

Requirements:
- OPENAI_API_KEY set in your environment
- ffmpeg installed and on PATH
- pip install -r requirements.txt
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import List

from tqdm import tqdm
import whisper
from openai import OpenAI


YOUTUBE_ID_PATTERNS = [
    r"(?:v=)([A-Za-z0-9_-]{11})",
    r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
    r"(?:/embed/)([A-Za-z0-9_-]{11})",
    r"(?:/shorts/)([A-Za-z0-9_-]{11})",
]

# Chunk output: clean Markdown, natural notes, detailed but not "worksheet-y".
CHUNK_PROMPT = """You are generating high-quality notes from a transcript slice.

Write clean Markdown that reads like real notes someone would keep.

Rules:
- Be detailed and specific, but do not invent facts.
- If the transcript is unclear, say so briefly (e.g., "unclear/garbled here").
- Keep the structure consistent and scannable.

Return ONLY this Markdown (no extra commentary):

### Summary
A short paragraph (3–6 sentences) capturing the main idea(s) and how the speaker develops them.

### Key Points
- 8–16 bullets with the main claims, reasoning steps, and important details.
- Prefer concrete wording over generic phrasing.

### Concepts & Definitions
- A compact list of terms + what they mean in this context.
- If there aren’t many terms, include fewer (quality > quantity).

### Examples / Analogies
- List concrete examples or analogies used and what they illustrate.
- If none: write "- None noted."

Transcript slice:
"""

# Final notes: exactly your headings, and more depth inside Detailed Notes.
FINAL_PROMPT = """You are synthesizing multiple chunk notes into one polished set of notes.

Output MUST be valid Markdown and include ONLY these top-level headings, in this exact order:

# Executive Summary
# Full Outline
# Detailed Notes
# Key Concepts & Definitions
# Memorable Examples / Analogies
# Action Items / Takeaways (if any)

Guidelines:
- Be detailed, but readable and not repetitive.
- Preserve the speaker's progression of ideas (early → middle → end).
- Do not include discussion/exam questions.
- Do not invent sources or add citations.
- If the content contains uncertainty or garbled parts, you may briefly note that.

Depth requirements:
- Executive Summary: 6–12 strong bullets (not generic).
- Full Outline: hierarchical outline with multiple levels where appropriate.
- Detailed Notes: the main body. Use subheadings, and include:
  - claims → support/evidence → implications
  - methods/processes/steps (if any)
  - tradeoffs, constraints, caveats (if any)
  - contrasting viewpoints or counterpoints (if present)
- Key Concepts & Definitions: clear, content-grounded definitions (alphabetize when reasonable).
- Memorable Examples / Analogies: include what each example was used to prove/clarify.
- Action Items / Takeaways: list explicit recommendations, practical steps, or "what to do next"; if none, write "None explicitly stated."

Here are the chunk notes to synthesize:
"""

def extract_video_id(url: str) -> str:
    for pat in YOUTUBE_ID_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError("Could not extract a YouTube video ID (must be 11 chars).")


def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH.\n"
            "Windows: winget install Gyan.FFmpeg\n"
            "Then reopen terminal and run: ffmpeg -version"
        )


def download_audio_with_ytdlp(url: str, out_template: str) -> str:
    """
    Downloads best audio using yt-dlp, invoked as a python module:
      python -m yt_dlp ...
    Returns the actual downloaded file path.
    """
    py = sys.executable

    cmd = [
        py, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "-o", out_template,
        url,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{p.stderr.strip() or p.stdout.strip()}")

    # Find the produced file (yt-dlp replaces %(ext)s)
    out_dir = os.path.dirname(out_template)
    base = os.path.basename(out_template).replace("%(ext)s", "")
    produced = [f for f in os.listdir(out_dir) if f.startswith(base)]
    if not produced:
        raise RuntimeError("yt-dlp ran but no audio file was produced.")

    produced_paths = [os.path.join(out_dir, f) for f in produced]
    produced_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return produced_paths[0]


def transcribe_with_whisper(audio_path: str, whisper_model: str = "base") -> str:
    model = whisper.load_model(whisper_model)
    result = model.transcribe(audio_path)
    return (result.get("text") or "").strip()


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        add = len(w) + 1
        if cur_len + add > max_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur = [w]
            cur_len = len(w) + 1
        else:
            cur.append(w)
            cur_len += add
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def call_openai(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.responses.create(model=model, input=prompt)
    return (resp.output_text or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("--out", default="notes.md", help="Output markdown file")
    ap.add_argument("--model", default="gpt-5.2", help="OpenAI model")
    ap.add_argument("--whisper-model", default="base", help="Whisper: tiny|base|small|medium|large")
    ap.add_argument("--max-chars", type=int, default=12000, help="Chunk size (approx chars)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            'ERROR: OPENAI_API_KEY is not set.\n'
            'PowerShell (temporary):  $env:OPENAI_API_KEY="your_key_here"\n'
            'PowerShell (permanent):  setx OPENAI_API_KEY "your_key_here"  (reopen terminal)',
            file=sys.stderr
        )
        return 2

    ensure_ffmpeg()

    # Validate video ID format early
    try:
        _ = extract_video_id(args.url)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    client = OpenAI()

    with tempfile.TemporaryDirectory() as td:
        out_template = os.path.join(td, "audio.%(ext)s")

        print("Downloading audio with yt-dlp (module)...")
        audio_file = download_audio_with_ytdlp(args.url, out_template)
        print(f"Downloaded: {audio_file}")

        print(f"Transcribing with Whisper ({args.whisper_model})...")
        transcript = transcribe_with_whisper(audio_file, whisper_model=args.whisper_model)

    if not transcript:
        print("ERROR: Transcript empty.", file=sys.stderr)
        return 3

    chunks = chunk_text(transcript, max_chars=args.max_chars)
    print(f"Transcript ready. Chunks: {len(chunks)}")

    chunk_notes = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks"), start=1):
        notes = call_openai(client, args.model, CHUNK_PROMPT + chunk)
        # Keep a clear divider so the final synthesis can track sequence.
        chunk_notes.append(f"## Chunk {i}\n\n{notes}".strip())

    print("Building final notes...")
    final_md = call_openai(
        client,
        args.model,
        FINAL_PROMPT + "\n\n---\n\n".join(chunk_notes)
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final_md.strip() + "\n")

    print(f"Done. Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
