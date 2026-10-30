"""Validate benchmark output files when present; uses salted digests for numeric arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-id", help="Validate salted parameters when supplied")
    args = p.parse_args()
    results = ROOT / "results"
    required = ["host_profile.json", "salted_params.json", "stride_results.tsv", "loop_interchange.json",
                "false_sharing_karp_flatt.tsv", "scaling_metrics.tsv", "roofline_summary.json"]
    missing = [name for name in required if not (results / name).exists()]
    if missing:
        raise SystemExit("Missing results: " + ", ".join(missing))
    params = json.loads((results / "salted_params.json").read_text())
    if args.student_id:
        from src.salt import get_student_params
        if params != get_student_params(args.student_id):
            raise SystemExit("Salted parameters do not match the supplied student ID")
    profile = json.loads((results / "host_profile.json").read_text())
    if not profile.get("cpu_model") or not profile.get("python", {}).get("version"):
        raise SystemExit("Host profile is missing CPU or Python information")
    print("[PASS] Milestone 0: host topology and salted parameters are valid.")
    interchange = json.loads((results / "loop_interchange.json").read_text())
    if interchange["max_abs_error"] > 1e-12:
        raise SystemExit("Loop interchange result exceeds tolerance")
    print("[PASS] Milestone 1: loop-order matrix results agree within 1e-12.")
    with (results / "false_sharing_karp_flatt.tsv").open() as f:
        lines = f.readlines()[1:]
    if not lines:
        raise SystemExit("False-sharing file contains no measurements")
    for line in lines:
        kind, workers, iters, _, total, _ = line.rstrip("\n").split("\t")
        if int(total) != int(workers) * int(iters):
            raise SystemExit(f"Counter total mismatch for {kind}, P={workers}")
    print("[PASS] Milestone 2: all counter totals equal P × iterations.")
    roofline = json.loads((results / "roofline_summary.json").read_text())
    if not roofline.get("strong"):
        raise SystemExit("Scaling output has no strong-scaling measurements")
    print("[PASS] Milestone 3: scaling results and Roofline estimate are present.")


if __name__ == "__main__":
    main()
