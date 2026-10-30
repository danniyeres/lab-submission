"""Measure read latency for different strides through a contiguous float64 array."""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salt import get_student_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--quick", action="store_true", help="Use a smaller buffer for a smoke run")
    args = parser.parse_args()
    count = get_student_params(args.student_id)["buffer_elements"]
    if args.quick:
        count = 1 << 20
    data = np.ones(count, dtype=np.float64)
    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        view = data[::stride]
        for _ in range(5):
            float(np.sum(view))
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            total = float(np.sum(view))
            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed / view.size)
        rows.append((stride, int(view.size), statistics.median(samples), total))
    path = Path(__file__).resolve().parents[1] / "results" / "stride_results.tsv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as f:
        f.write("stride\telements_touched\tmedian_ns_per_element\tchecksum\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    print(path)


if __name__ == "__main__":
    main()
