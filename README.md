# YouTube Video Summarizer with Whisper and LLMs

## Overview
This project converts YouTube videos into structured, high-quality study notes.
It automatically downloads audio, transcribes speech using OpenAI Whisper, and
generates detailed Markdown notes using a large language model.

The tool is intended for students, researchers, and professionals who want
organized insights from long-form video content such as lectures, talks,
podcasts, and interviews.

## Pipeline
1. Download audio from a YouTube URL
2. Transcribe speech using OpenAI Whisper
3. Chunk long transcripts for efficient processing
4. Generate section summaries using an LLM
5. Produce polished Markdown notes

## Requirements
- Python 3.9 or newer
- ffmpeg installed and available on PATH
- An OpenAI API key (users provide their own)

## Setup
Install dependencies:
pip install -r requirements.txt

Verify ffmpeg:
ffmpeg -version

Set your OpenAI API key (PowerShell, current session):
$env:OPENAI_API_KEY="your_key_here"

## Usage
python ytsum.py "<youtube_url>" --output notes.md

Example:
python ytsum.py "<youtube_url>" --output notes.md

## Output
The program generates a Markdown file containing the following sections:
- Executive Summary
- Full Outline
- Detailed Notes
- Key Concepts & Definitions
- Memorable Examples / Analogies
- Action Items / Takeaways (if any)

## Technologies
Python, OpenAI Whisper, OpenAI API, yt-dlp, ffmpeg
