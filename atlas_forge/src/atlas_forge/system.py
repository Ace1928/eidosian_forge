import psutil
import logging
from typing import Any, Dict
from .config import FORGE_ROOT

logger = logging.getLogger("eidos_dashboard")

def get_system_stats() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cpu": None,
        "ram_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "disk_percent": None,
        "disk_free_gb": None,
        "uptime": None
    }
    
    # CPU Usage
    try:
        # psutil.cpu_percent with interval=None is non-blocking but might need 
        # a previous call to be accurate. We'll just try to get it.
        payload["cpu"] = psutil.cpu_percent(interval=None)
    except Exception as e:
        logger.warning(f"Unable to read CPU percent: {e}")

    # Memory Usage
    try:
        mem = psutil.virtual_memory()
        payload.update({
            "ram_percent": mem.percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
        })
    except Exception as e:
        logger.warning(f"Unable to read memory stats: {e}")

    # Disk Usage
    try:
        disk = psutil.disk_usage(str(FORGE_ROOT))
        payload.update({
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
        })
    except Exception as e:
        logger.warning(f"Unable to read disk stats: {e}")

    # Uptime
    try:
        payload["uptime"] = int(psutil.boot_time())
    except Exception as e:
        logger.warning(f"Unable to read boot time: {e}")
        
    return payload
