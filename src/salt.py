"""Deterministic student-specific benchmark parameters."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def get_student_params(student_id: str) -> dict:
    if not re.fullmatch(r"\d{8,}", student_id):
        raise ValueError("student_id must contain at least 8 digits")
    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    return {
        "student_id": student_id,
        "N": 1024 + (h % 9) * 64,
        "buffer_elements": 16_777_216,
        "seed": h % 100_007,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "results" / "salted_params.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(get_student_params(args.student_id), indent=2) + "\n")
    print(output)
