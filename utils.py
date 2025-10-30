import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import time

def extract_frames(video_path, out_dir="frames"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    idx = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(fname, frame)
        frames.append(fname)
        idx += 1
    cap.release()
    elapsed = time.time() - start
    return frames, fps, (w, h), elapsed

def load_frame(path, resize=None):
    img = cv2.imread(path)
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return img

def color_histogram(img, mask=None, bins=(32, 32, 32)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def hist_similarity(h1, h2):
    return (cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL) + 1)/2.0

def ssim_gray(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    val = ssim(ga, gb, data_range=ga.max() - ga.min())
    return float(val)

def orb_match_score(a, b, nfeatures=500):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    score = len(matches) / max(1, min(len(kp1), len(kp2)))
    return min(1.0, score)

def combined_similarity(a, b, hist_weight=0.4, ssim_weight=0.4, orb_weight=0.2,
                        hist_cache=None, emb_cache=None):
    h1 = color_histogram(a) if hist_cache is None else hist_cache
    h2 = color_histogram(b) if hist_cache is None else hist_cache
    hs = hist_similarity(h1, h2)
    ss = ssim_gray(a, b)
    oscore = orb_match_score(a, b)
    sim = hist_weight*hs + ssim_weight*ss + orb_weight*oscore
    return max(0.0, min(1.0, sim))
