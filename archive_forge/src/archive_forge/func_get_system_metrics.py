import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
def get_system_metrics() -> Dict[str, Union[float, dict]]:
    """Get CPU and other performance metrics."""
    global _PSUTIL_AVAILABLE
    if not _PSUTIL_AVAILABLE:
        return {}
    try:
        process = psutil.Process(os.getpid())
        metrics: Dict[str, Union[float, dict]] = {}
        with process.oneshot():
            mem_info = process.memory_info()
            metrics['thread_count'] = float(process.num_threads())
            metrics['mem'] = {'rss': float(mem_info.rss)}
            ctx_switches = process.num_ctx_switches()
            cpu_times = process.cpu_times()
            metrics['cpu'] = {'time': {'sys': cpu_times.system, 'user': cpu_times.user}, 'ctx_switches': {'voluntary': float(ctx_switches.voluntary), 'involuntary': float(ctx_switches.involuntary)}, 'percent': process.cpu_percent()}
        return metrics
    except Exception as e:
        _PSUTIL_AVAILABLE = False
        logger.debug('Failed to get system metrics: %s', e)
        return {}