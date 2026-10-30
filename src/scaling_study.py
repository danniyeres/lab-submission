"""Strong/weak scaling of a process-parallel 2D stencil, with roofline estimate."""
import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

import numpy as np


def _stencil_chunk(args):
    array, start, stop = args
    # Five-point stencil over assigned interior rows.
    return (array[start-1:stop-1, 1:-1] + array[start+1:stop+1, 1:-1]
            + array[start:stop, :-2] + array[start:stop, 2:]
            + array[start:stop, 1:-1]) * 0.2


def _run(array, workers):
    n = array.shape[0] - 2
    edges = np.linspace(1, n + 1, workers + 1, dtype=int)
    tasks = [(array, int(edges[i]), int(edges[i+1])) for i in range(workers)]
    with mp.Pool(workers) as pool:
        return pool.map(_stencil_chunk, tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--size", type=int, help="Override salted N for a practical run")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    from salt import get_student_params
    params = get_student_params(args.student_id)
    n = args.size or params["N"]
    if args.quick:
        n = min(n, 256)
    maxp = args.max_workers or (os.cpu_count() or 1)
    workers = sorted({1, 2, 4, 8, maxp})
    workers = [p for p in workers if p <= maxp]
    rng = np.random.default_rng(params["seed"])
    array = rng.random((n + 2, n + 2), dtype=np.float64)
    reference = _stencil_chunk((array, 1, n + 1))
    strong = {}
    strong_times = {}
    for p in workers:
        samples = []
        answer = None
        for _ in range(3):
            start = time.perf_counter()
            answer = _run(array, p)
            samples.append(time.perf_counter() - start)
        joined = np.concatenate(answer, axis=0)
        if not np.allclose(joined, reference, rtol=0, atol=1e-15):
            raise AssertionError("strong-scaling stencil differs from serial reference")
        strong_times[p] = statistics.median(samples)
    t1 = strong_times[1]
    for p in workers:
        speedup = t1 / strong_times[p]
        serial = 0.0 if p == 1 else max(0.0, (1/speedup - 1/p) / (1 - 1/p))
        strong[p] = {"seconds": strong_times[p], "speedup": speedup,
                     "efficiency_percent": speedup/p*100, "karp_flatt_serial_fraction": serial}
    weak = {}
    for p in workers:
        np_ = math.ceil(n * math.sqrt(p))
        a = rng.random((np_ + 2, np_ + 2), dtype=np.float64)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            _run(a, p)
            samples.append(time.perf_counter() - start)
        t = statistics.median(samples)
        # Gustafson scaled speedup compares normalized single-worker time to the
        # weak-scaling run: ideal time for the larger problem grows by P.
        weak[p] = {"dimension": np_, "seconds": t,
                   "scaled_speedup": (p * strong_times[1]) / t}
    outdir = Path(__file__).resolve().parents[1] / "results"
    with (outdir / "scaling_metrics.tsv").open("w") as f:
        f.write("study\tprocesses\tdimension\tseconds\tspeedup_or_scaled_speedup\tefficiency_percent\tserial_fraction\n")
        for p in workers:
            f.write(f"strong\t{p}\t{n}\t{strong[p]['seconds']:.9f}\t{strong[p]['speedup']:.6f}\t{strong[p]['efficiency_percent']:.4f}\t{strong[p]['karp_flatt_serial_fraction']:.8f}\n")
            f.write(f"weak\t{p}\t{weak[p]['dimension']}\t{weak[p]['seconds']:.9f}\t{weak[p]['scaled_speedup']:.6f}\tNA\tNA\n")
    ops = 5 * n * n
    bytes_est = 6 * 8 * n * n
    roofline = {"kernel": "five-point average stencil", "strong_dimension": n,
                "floating_point_ops_per_cell": 5, "estimated_dram_bytes_per_cell": 48,
                "arithmetic_intensity_flop_per_byte": ops / bytes_est,
                "estimated_peak_memory_bandwidth_bytes_per_second": None,
                "classification": "undetermined_without_platform_peak_bandwidth_and_measured_kernel_flops",
                "note": "The 48 B/cell model assumes five float64 loads and one float64 store; cache reuse can reduce DRAM traffic."}
    (outdir / "roofline_summary.json").write_text(json.dumps({"strong": strong, "weak": weak, "roofline": roofline}, indent=2) + "\n")
    print(outdir / "scaling_metrics.tsv")


if __name__ == "__main__":
    main()
