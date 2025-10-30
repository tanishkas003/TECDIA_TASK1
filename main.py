import argparse
import os
import time
import numpy as np
import cv2
from utils import extract_frames, load_frame, combined_similarity, color_histogram
from joblib import Parallel, delayed
from tqdm import tqdm
from solver import multi_start_greedy
import json

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def compute_pairwise_similarities(frame_paths, mode='basic', workers=8, resize_for_compute=(360, 202)):
    """Compute pairwise frame similarities using histograms, SSIM, ORB, and optional CNN embeddings."""
    N = len(frame_paths)
    frames = [load_frame(p, resize=resize_for_compute) for p in frame_paths]
    hists = [color_histogram(f) for f in frames]
    emb = None

    if mode == 'advanced' and TORCH_AVAILABLE:
        emb = compute_cnn_embeddings(frames)

    def compute_row(i):
        row = np.zeros(N, dtype=np.float32)
        a = frames[i]
        for j in range(N):
            if i == j:
                row[j] = 1.0
                continue
            b = frames[j]

            if emb is not None:
                v1, v2 = emb[i], emb[j]
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
                cos_s = (cos + 1) / 2.0
                ss_val = combined_similarity(a, b)
                hs = (cv2.compareHist(hists[i].astype('float32'), hists[j].astype('float32'), cv2.HISTCMP_CORREL) + 1) / 2.0
                final = 0.6 * cos_s + 0.3 * hs + 0.1 * ss_val
                row[j] = final
            else:
                final = combined_similarity(a, b, hist_weight=0.45, ssim_weight=0.45, orb_weight=0.10)
                row[j] = final
        return row

    results = Parallel(n_jobs=workers)(delayed(compute_row)(i) for i in tqdm(range(N), desc="Computing similarities"))
    sim_mat = np.stack(results, axis=0)
    return sim_mat

def compute_cnn_embeddings(frames, device='cpu', batch_size=32):
    model = models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    feat_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tbatch = torch.stack([transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch]).to(device)
            feats = feat_extractor(tbatch).squeeze(-1).squeeze(-1)
            feats = feats.cpu().numpy()
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10)
            embeddings.append(feats)
    return np.vstack(embeddings)

def sim_to_cost(sim_mat, epsilon=1e-6):
    cost = 1.0 - sim_mat
    np.fill_diagonal(cost, 1.0)
    cost = np.clip(cost, epsilon, None)
    return cost


def make_video_from_order(frame_paths, order, out_path, fps=30, size=None):
    if size is None:
        img = cv2.imread(frame_paths[0])
        h, w = img.shape[:2]
        size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    for idx in order:
        frame = cv2.imread(frame_paths[idx])
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    writer.release()

def decide_direction(order, frame_paths):
    """
    Detects reversed segments using optical flow direction consistency.
    If the first few seconds move opposite to main motion trend, flips that section.
    """
    import numpy as np
    import cv2

    print("Checking direction consistency using optical flow...")

    def flow_mean_direction(img1, img2):
        g1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return np.mean(flow[..., 0])  

    flow_dirs = []
    for i in range(len(order) - 1):
        try:
            val = flow_mean_direction(frame_paths[order[i]], frame_paths[order[i + 1]])
            flow_dirs.append(val)
        except Exception:
            flow_dirs.append(0)

    flow_dirs = np.array(flow_dirs)
    median_dir = np.median(flow_dirs[len(flow_dirs) // 2:])  

    reversed_indices = np.where(np.sign(flow_dirs[:40]) != np.sign(median_dir))[0]

    if len(reversed_indices) > 10:
        split_point = reversed_indices[-1] + 1
        print(f"Reversed motion detected in first {split_point} frames — flipping that section.")
        order = order[:split_point][::-1] + order[split_point:]
    else:
        print("Direction looks correct. No reversed section found.")

    return order

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='path to jumbled video')
    parser.add_argument('--out', type=str, default='outputs/reconstructed_final.mp4')
    parser.add_argument('--mode', type=str, default='basic', choices=['basic', 'advanced'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--frames_dir', type=str, default='frames')
    parser.add_argument('--log', type=str, default='logs/timing_log.txt')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    total_start = time.time()
    print("\n[1/5] Extracting frames...")
    frames, video_fps, size, t_extract = extract_frames(args.video, out_dir=args.frames_dir)
    fps = args.fps if args.fps else video_fps
    print(f"Extracted {len(frames)} frames ({fps} FPS) in {t_extract:.2f}s")

    print("\n[2/5] Computing pairwise similarities (this may take time)...")
    t0 = time.time()
    sim_mat = compute_pairwise_similarities(frames, mode=args.mode, workers=args.workers)
    t_sim = time.time() - t0
    print(f"Similarity matrix computed in {t_sim:.2f}s")

    print("\n[3/5] Solving best frame order...")
    cost_mat = sim_to_cost(sim_mat)
    t0 = time.time()
    order, best_cost = multi_start_greedy(cost_mat, starts=20)
    order = decide_direction(order, frames)
    t_solver = time.time() - t0
    print(f"Solved optimal order in {t_solver:.2f}s | cost={best_cost:.4f}")

    print("\n[4/5] Writing reconstructed video...")
    t0 = time.time()
    make_video_from_order(frames, order, args.out, fps=int(fps), size=size)
    t_write = time.time() - t0
    print(f"Video written to {args.out} in {t_write:.2f}s")

    total_elapsed = time.time() - total_start
    log = {
        'video': args.video,
        'num_frames': len(frames),
        'fps': fps,
        'mode': args.mode,
        'workers': args.workers,
        'times': {
            'frame_extraction': t_extract,
            'similarity_comp': t_sim,
            'solver': t_solver,
            'write': t_write,
            'total': total_elapsed
        },
        'solver_cost': float(best_cost)
    }

    with open(args.log, 'w') as f:
        json.dump(log, f, indent=2)
    print("\nAll done! Timing log saved to", args.log)
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
