#!/usr/bin/env python3
import argparse
import os
import pickle
import numpy as np


def load_concept_vectors(data_dir, split):
    pkl_path = os.path.join(data_dir, f"{split}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    by_class = {}
    for d in data:
        cls = d["class_label"]
        vec = np.array(d["attribute_label"])
        by_class.setdefault(cls, []).append(vec)
    return by_class


def main():
    parser = argparse.ArgumentParser(description="Extract concept vectors from CUB .pkl files")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/val/test .pkl files")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which split to load")
    parser.add_argument("--out", default=None, help="Optional .npz output path")
    args = parser.parse_args()

    by_class = load_concept_vectors(args.data_dir, args.split)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        # Save as compressed npz with one array per class
        np.savez_compressed(args.out, **{str(k): np.stack(v, axis=0) for k, v in by_class.items()})
        print(args.out)
    else:
        sizes = {k: len(v) for k, v in by_class.items()}
        print(f"Loaded {len(by_class)} classes")
        print(f"Example sizes: {list(sizes.items())[:5]}")


if __name__ == "__main__":
    main()
