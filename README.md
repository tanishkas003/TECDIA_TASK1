# Jumbled Frames Reconstruction Challenge - Reference Solution

## Overview
This repo reconstructs a shuffled single-shot video (10s, 300 frames) by computing pairwise frame similarities and solving for a likely frame order.

## Requirements
Python 3.8+.

Install dependencies:
pip install -r requirements.txt

## Files
- `main.py`: pipeline to extract frames, compute similarities, solve order and write reconstructed video.
- `utils.py`: image similarity utilities.
- `solver.py`: greedy + 2-opt solver.
- `requirements.txt`

## Usage

Basic (fast, good accuracy):python main.py --video path/to/jumbled_video.mp4 --out outputs/reconstructed_basic.mp4 --mode basic --workers 8

Notes:
- `--workers`: controls parallelism for similarity matrix computation.
- Execution time log and metrics are written to `logs/timing_log.txt` (JSON).

## Output
- `outputs/reconstructed*.mp4` — reconstructed video
- `logs/timing_log.txt` — timing and solver summary

## How the solver works
1. Build similarity matrix between every ordered pair of frames.
2. Convert to cost = 1 - similarity.
3. Use multi-start greedy to produce initial paths and apply 2-opt local search to refine.
4. Pick best found path and produce output video.

## Tips for usage
- Start with `--mode basic` and `--workers` equal to CPU cores minus 1.
- If you use `--mode advanced`, allow extra time (embedding + pairwise compute).

### Execution Time
| Step | Duration (seconds) |
|------|--------------------|
| Frame extraction | 22.01410722732544 |
| Similarity computation | 643.5727665424347 |
| Solver | 808.0119268894196 |
| Video writing | 13.531448125839233 |
| **Total** | **1487.2949388027191** |

### reconstructed_final.mp4 is the final result

### the reconstructed_basic.mp4 was the first trial where there was an issue with my code because of which a first few seconds of the video were in reverse so fixed that

## 🎥 Reconstructed Video
[Watch on Google Drive](https://drive.google.com/file/d/1UDRH4qE0GpN8nZj9tvgI5lpKCOimUHe8/view?usp=sharing)

## 🧾 Execution Log
[View timing_log.txt](https://drive.google.com/file/d/1WmEa1IPjqupaw41BRI3cuyBUTPMcbd5i/view?usp=sharing)