"""Pure Python loop-order comparison, using the salted N/2 submatrix."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from salt import get_student_params


def multiply_ijk(a, b):
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += a[i][k] * b[k][j]
            c[i][j] = acc
    return c


def multiply_ikj(a, b):
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ci = c[i]
        for k in range(n):
            aik = a[i][k]
            bk = b[k]
            for j in range(n):
                ci[j] += aik * bk[j]
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-id", required=True)
    p.add_argument("--size", type=int, help="Override sub-block size for practical local runs")
    args = p.parse_args()
    params = get_student_params(args.student_id)
    n = args.size or params["N"] // 2
    rng = np.random.default_rng(params["seed"])
    a = rng.random((n, n)).tolist()
    b = rng.random((n, n)).tolist()
    # A single pure-Python 512^3 run is already expensive; three samples follow the lab spec.
    results = {}
    outputs = {}
    for name, fn in (("ijk", multiply_ijk), ("ikj", multiply_ikj)):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            outputs[name] = fn(a, b)
            times.append(time.perf_counter() - start)
        results[name] = {"median_seconds": statistics.median(times), "samples_seconds": times}
    x = np.asarray(outputs["ijk"])
    y = np.asarray(outputs["ikj"])
    results.update({"N": params["N"], "sub_block_size": n,
                    "speedup_ijk_over_ikj": results["ijk"]["median_seconds"] / results["ikj"]["median_seconds"],
                    "max_abs_error": float(np.max(np.abs(x-y))),
                    "reference_sha256_rounded_6": __import__("hashlib").sha256(np.round(x, 6).astype("<f8").tobytes()).hexdigest()})
    path = Path(__file__).resolve().parents[1] / "results" / "loop_interchange.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
