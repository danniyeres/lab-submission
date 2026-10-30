import argparse
import ctypes
import multiprocessing as mp
import statistics
import time
import json
from pathlib import Path


def _worker_unpadded(shared, worker_id, iterations):
    for _ in range(iterations):
        shared[worker_id] += 1


def _worker_padded(shared, worker_id, iterations, stride):
    index = worker_id * stride
    for _ in range(iterations):
        shared[index] += 1


def _worker_local(shared, worker_id, iterations):
    value = 0
    for _ in range(iterations):
        value += 1
    shared[worker_id] = value


def run(kind, workers, iterations, stride):
    n = workers * stride if kind == "padded" else workers
    shared = mp.Array(ctypes.c_int64, n, lock=False)
    worker = {"unpadded": _worker_unpadded, "padded": _worker_padded, "privatized": _worker_local}[kind]
    procs = [mp.Process(target=worker, args=(shared, i, iterations, stride) if kind == "padded"
                        else (shared, i, iterations)) for i in range(workers)]
    start = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - start
    if any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError(f"worker failure in {kind} with P={workers}")
    return elapsed, sum(shared[i * stride if kind == "padded" else i] for i in range(workers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5_000_000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    iterations = 100_000 if args.quick else args.iterations
    profile_path = Path(__file__).resolve().parents[1] / "results" / "host_profile.json"
    detected_line = json.loads(profile_path.read_text()).get("detected_cache_line_bytes") if profile_path.exists() else None
    stride = max(8, (int(detected_line or 64) + 7) // 8)
    records = {}
    for kind in ("unpadded", "padded", "privatized"):
        for p in (1, 2, 4, 8):
            samples = []
            total = None
            for _ in range(3):
                elapsed, total = run(kind, p, iterations, stride)
                samples.append(elapsed)
            if total != p * iterations:
                raise AssertionError(f"counter mismatch: {total} != {p * iterations}")
            records[(kind, p)] = statistics.median(samples)
    out = Path(__file__).resolve().parents[1] / "results" / "false_sharing_karp_flatt.tsv"
    with out.open("w") as f:
        f.write("implementation\tprocesses\titerations_per_process\tmedian_seconds\ttotal_count\tkarp_flatt_e\n")
        baseline = {kind: records[(kind, 1)] for kind in ("unpadded", "padded", "privatized")}
        for kind in ("unpadded", "padded", "privatized"):
            for p in (1, 2, 4, 8):
                e = "NA" if p == 1 else (p * records[(kind, p)] / baseline[kind] - 1) / (p - 1)
                f.write(f"{kind}\t{p}\t{iterations}\t{records[(kind,p)]:.9f}\t{p*iterations}\t{e}\n")
    print(out)


if __name__ == "__main__":
    main()
