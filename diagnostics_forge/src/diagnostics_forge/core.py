from __future__ import annotations
import os
import time
import psutil
import subprocess
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from eidosian_core import eidosian

class DiagnosticsForge:
    """
    The adaptive observability engine for Eidos.
    Provides parity between standard Linux and restricted Termux environments.
    """

    def __init__(self, service_name: str = "core"):
        self.service_name = service_name
        self.start_time = time.time()

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
