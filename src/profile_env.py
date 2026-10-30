"""Collect best-effort host telemetry on macOS and Linux."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    import multiprocessing
except ImportError:  # pragma: no cover
    multiprocessing = None


def sysctl_value(name):
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def profile():
    system = platform.system()
    cpu = platform.processor() or platform.machine()
    physical = logical = None
    caches = {"L1_data_bytes": None, "L2_bytes": None, "L3_bytes": None, "cache_line_bytes": None}
    if system == "Darwin":
        cpu = sysctl_value("machdep.cpu.brand_string") or _mac_cpu_name() or cpu
        physical = _as_int(sysctl_value("hw.physicalcpu"))
        logical = _as_int(sysctl_value("hw.logicalcpu"))
        caches.update({
            "L1_data_bytes": _as_int(sysctl_value("hw.l1dcachesize")),
            "L2_bytes": _as_int(sysctl_value("hw.l2cachesize")),
            "L3_bytes": _as_int(sysctl_value("hw.l3cachesize")),
            "cache_line_bytes": _as_int(sysctl_value("hw.cachelinesize")),
        })
        if not logical:
            logical = os.cpu_count()
        if not physical:
            physical = logical
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace") if Path("/proc/cpuinfo").exists() else ""
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu = line.split(":", 1)[-1].strip()
                break
        logical = os.cpu_count()
        physical = logical
        cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
        for index in cache_root.glob("index*"):
            try:
                level = (index / "level").read_text().strip()
                size = (index / "size").read_text().strip()
                line_size = int((index / "coherency_line_size").read_text().strip())
                amount = int(size[:-1]) * (1024 if size.endswith("K") else 1)
                key = {"1": "L1_data_bytes", "2": "L2_bytes", "3": "L3_bytes"}.get(level)
                if key and (key != "L1_data_bytes" or (index / "type").read_text().strip() == "Data"):
                    caches[key] = amount
                    caches["cache_line_bytes"] = line_size
            except (OSError, ValueError):
                pass
    else:
        logical = os.cpu_count()
    return {
        "os": platform.platform(), "kernel_version": platform.release(),
        "cpu_model": cpu, "microarchitecture": platform.machine(),
        "base_frequency_hz": _as_int(sysctl_value("hw.cpufrequency")) if system == "Darwin" else None,
        "physical_cores": physical, "logical_threads": logical,
        "smt_or_hyperthreading": bool(physical and logical and logical > physical),
        "caches": caches,
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "numpy_version": _numpy_version(),
        "notes": "Unavailable telemetry is null; macOS sysctl values depend on host permissions.",
    }


def _as_int(value):
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _mac_cpu_name():
    try:
        result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                                text=True, timeout=12)
        for line in result.stdout.splitlines():
            if "Chip:" in line or "Processor Name:" in line:
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _numpy_version():
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from salt import get_student_params
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "host_profile.json").write_text(json.dumps(profile(), indent=2) + "\n")
    (root / "results" / "salted_params.json").write_text(json.dumps(get_student_params(args.student_id), indent=2) + "\n")
