#!/usr/bin/env python3
"""
Incremental PCA on a folder of RGB images with varying sizes.

- First pass: compute average H and W across all readable images
- Second pass: stream images, resize to HxW, and feed batches into IncrementalPCA
- Saves: mean.png (optional), first K components visualized, explained_variance.csv

Run as a script:
    python eigen_images.py --input INPUT_DIR --output OUTPUT_DIR --k 16 --batch 32 --save_mean
"""
import os
import argparse
from glob import glob
from typing import List, Tuple, Iterable
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.decomposition import IncrementalPCA
except Exception as e:
    raise SystemExit(
        "scikit-learn is required. Please install it, for example:\n"
        "  pip install scikit-learn\n"
        f"Import error was: {e}"
    )

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))

def compute_avg_hw(paths: List[str]) -> Tuple[int, int, List[str]]:
    hs, ws = [], []
    good = []
    for p in tqdm(paths, desc="Scanning sizes", unit="img"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                h, w = im.height, im.width
                hs.append(h); ws.append(w)
                good.append(p)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    if not hs:
        raise RuntimeError("No readable images to compute average size.")
    H = int(round(float(np.mean(hs)))); H = max(H, 1)
    W = int(round(float(np.mean(ws)))); W = max(W, 1)
    return H, W, good

def load_batch(paths: List[str], H: int, W: int) -> np.ndarray:
    """Load a batch into [B, H*W*3] float32 in [0,1]."""
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.size != (W, H):
                    im = im.resize((W, H), resample=Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                out.append(arr.reshape(-1))
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")
    if not out:
        return np.empty((0, H*W*3), dtype=np.float32)
    return np.stack(out, axis=0)

def batches(paths: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(paths), batch_size):
        yield paths[i:i+batch_size]

def save_image(arr01: np.ndarray, path: str) -> None:
    arr = np.clip(arr01 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)

def visualize_component(vec: np.ndarray, H: int, W: int) -> np.ndarray:
    e = vec.reshape(H, W, 3)
    vmin = e.min(); vmax = e.max()
    if vmax - vmin < 1e-12:
        return np.zeros((H, W, 3), dtype=np.float32)
    vis = (e - vmin) / (vmax - vmin)
    return np.clip(vis, 0.0, 1.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with RGB images")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--k", type=int, default=16, help="Number of eigenvectors to render")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for streaming")
    ap.add_argument("--save_mean", action="store_true", help="Save the dataset mean image")
    ap.add_argument("--prefix", type=str, default="eig", help="Filename prefix for eigen images")
    args = ap.parse_args()

    in_dir = args.input
    out_dir = args.output
    k = max(1, int(args.k))
    bs = max(1, int(args.batch))

    os.makedirs(out_dir, exist_ok=True)

    all_paths = list_images(in_dir)
    if not all_paths:
        raise SystemExit(f"No image files found in {in_dir}")

    # Pass 1: sizes
    H, W, good_paths = compute_avg_hw(all_paths)
    print(f"Average size -> H={H}, W={W}")
    N = len(good_paths)
    D = H * W * 3
    print(f"Estimated data matrix: N={N}, D={D}")

    # Build IncrementalPCA
    n_components = min(k, N, D)
    if n_components < k:
        print(f"[info] reducing k from {k} to {n_components} due to data limits")
    ipca = IncrementalPCA(n_components=n_components, batch_size=bs)

    # Pass 2: partial_fit over batches
    for batch_paths in tqdm(list(batches(good_paths, bs)), desc="Fitting IPCA", unit="batch"):
        Xb = load_batch(batch_paths, H, W)  # [B, D]
        if Xb.shape[0] == 0:
            continue
        ipca.partial_fit(Xb)

    # Components and stats
    comps = ipca.components_                # [k,D]
    ev = ipca.explained_variance_           # [k]
    evr = ipca.explained_variance_ratio_    # [k]
    mean_vec = ipca.mean_                    # [D]

    # Save outputs
    import csv
    if args.save_mean:
        mean_img = mean_vec.reshape(H, W, 3)
        save_image(np.clip(mean_img, 0.0, 1.0), os.path.join(out_dir, "mean.png"))
        print("Saved mean.png")

    for i in tqdm(range(n_components), desc="Saving eigen images", unit="img"):
        vis = visualize_component(comps[i], H, W)
        save_image(vis, os.path.join(out_dir, f"{args.prefix}_{i+1:02d}.png"))

    csv_path = os.path.join(out_dir, "explained_variance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "explained_variance", "explained_variance_ratio"])
        for i in range(n_components):
            writer.writerow([i+1, float(ev[i]), float(evr[i])])
    print(f"Wrote explained variance to {csv_path}")

if __name__ == "__main__":
    main()
