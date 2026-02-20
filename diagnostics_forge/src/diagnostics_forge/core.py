from __future__ import annotations
import json
import logging
import os
import time
import psutil
import subprocess
import re
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional
from pathlib import Path
from eidosian_core import eidosian

class DiagnosticsForge:
    """
    The adaptive observability engine for Eidos.
    Provides parity between standard Linux and restricted Termux environments.
    """

    def __init__(
        self,
        service_name: str = "core",
        log_dir: str | Path | None = None,
        json_format: bool = False,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.service_name = service_name
        self.start_time = time.time()
        self.log_dir = Path(log_dir) if log_dir is not None else (Path.cwd() / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{self.service_name}.log"
        self.metrics_file = self.log_dir / f"{self.service_name}_metrics.json"
        self.json_format = json_format
        self._timers: dict[str, float] = {}
        self._metrics: dict[str, list[float]] = {}
        logger_key = str(self.log_file).replace(os.sep, "_")
        self._logger = logging.getLogger(f"diagnostics_forge.{self.service_name}.{logger_key}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @eidosian()
    def log_event(self, level: str, message: str, **data: Any) -> None:
        """Log a structured diagnostics event."""
        level_name = (level or "INFO").upper()
        level_num = getattr(logging, level_name, logging.INFO)
        if self.json_format:
            payload: dict[str, Any] = {"level": level_name, "message": message}
            payload.update(data)
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            return

        suffix = ""
        if data:
            suffix = f' DATA: {json.dumps(data, ensure_ascii=True, sort_keys=True)}'
        self._logger.log(level_num, f"{message}{suffix}")

    @eidosian()
    def start_timer(self, metric_name: str) -> str:
        timer_id = f"{metric_name}:{time.time_ns()}"
        self._timers[timer_id] = time.perf_counter()
        return timer_id

    @eidosian()
    def stop_timer(self, timer_id: str) -> float:
        start = self._timers.pop(timer_id, None)
        if start is None:
            return 0.0
        duration = time.perf_counter() - start
        metric_name = timer_id.split(":", 1)[0]
        self._metrics.setdefault(metric_name, []).append(duration)
        return duration

    @eidosian()
    def get_metrics_summary(self, metric_name: str) -> Dict[str, Any]:
        values = self._metrics.get(metric_name, [])
        if not values:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    @eidosian()
    def save_metrics(self, file_path: str | Path | None = None) -> Path:
        out = Path(file_path) if file_path is not None else self.metrics_file
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: self.get_metrics_summary(name) for name in sorted(self._metrics)}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return out

    def _get_cpu_fallback(self) -> float:
        """Fallback for CPU usage using 'top'."""
        try:
            # -n 1 for one iteration, -b for batch mode
            result = subprocess.check_output(["top", "-n", "1", "-b"], stderr=subprocess.STDOUT).decode()
            
            # Pattern 1: Termux 'top' (User %, Nice %, System %, etc.)
            match = re.search(r"User\s+(\d+)%", result)
            if match:
                user_pct = float(match.group(1))
                sys_match = re.search(r"System\s+(\d+)%", result)
                sys_pct = float(sys_match.group(1)) if sys_match else 0.0
                return user_pct + sys_pct

            # Pattern 2: Standard Linux 'top'
            match = re.search(r"%Cpu\(s\):\s+([\d\.]+)\s+us", result)
            if match:
                return float(match.group(1))
            
            return 0.0
        except Exception:
            return 0.0

    def _get_load_fallback(self) -> List[float]:
        """Fallback for Load Avg using 'uptime'."""
        try:
            result = subprocess.check_output(["uptime"]).decode()
            parts = re.split(r"load average[s]?:\s+", result)
            if len(parts) > 1:
                nums = re.findall(r"[\d\.]+", parts[1])
                return [float(n) for n in nums[:3]]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    @eidosian()
    def get_system_pulse(self) -> Dict[str, Any]:
        """Collect current system resource metrics with adaptive fallbacks."""
        # CPU Percent
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except (PermissionError, Exception):
            cpu_percent = self._get_cpu_fallback()

        # Load Average
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError, PermissionError):
            load_avg = self._get_load_fallback()

        # Memory
        try:
            mem = psutil.virtual_memory()
            memory_data = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent
            }
        except Exception:
            memory_data = {"error": "Memory metrics unavailable"}

        # Disk
        try:
            disk = psutil.disk_usage('/')
            disk_data = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent": disk.percent,
                "note": "Disk usage figures may be inaccurate in Termux/Android environments."
            }
        except Exception:
            disk_data = {"error": "Disk metrics unavailable"}
        
        return {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "load_avg": load_avg
            },
            "memory": memory_data,
            "disk": disk_data,
            "environment": "termux" if "/com.termux/" in os.environ.get("PATH", "") else "linux"
        }

    @eidosian()
    def get_process_metrics(self) -> List[Dict[str, Any]]:
        """Identify and monitor key Eidosian processes with fallback safety."""
        eidos_procs = []
        keywords = ["eidos", "llama", "python", "ollama", "node"]
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                info = proc.info
                cmdline = " ".join(info['cmdline'] or [])
                if any(kw in cmdline.lower() or kw in (info['name'] or "").lower() for kw in keywords):
                    if "grep" in cmdline or "ps -ef" in cmdline:
                        continue
                        
                    eidos_procs.append({
                        "pid": info['pid'],
                        "name": info['name'],
                        "cpu": info.get('cpu_percent', 0.0),
                        "memory_mb": round(info['memory_info'].rss / (1024 * 1024), 2) if info.get('memory_info') else 0.0,
                        "cmd": cmdline[:100] + "..." if len(cmdline) > 100 else cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
        
        return sorted(eidos_procs, key=lambda x: x['cpu'], reverse=True)
