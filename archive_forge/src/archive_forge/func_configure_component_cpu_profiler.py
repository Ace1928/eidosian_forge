import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def configure_component_cpu_profiler(component_name: str, component_id: str, component_type: Optional[ServeComponentType]=None) -> Tuple[Optional[cProfile.Profile], Optional[str]]:
    """Configures the CPU profiler for this component.

    Does nothing if RAY_SERVE_ENABLE_CPU_PROFILING is disabled.

    Returns:
        2-tuple containing profiler object and log file name for profile stats.
    """
    if RAY_SERVE_ENABLE_CPU_PROFILING:
        logger = logging.getLogger(SERVE_LOGGER_NAME)
        try:
            import cProfile
        except ImportError:
            logger.warning('RAY_SERVE_ENABLE_CPU_PROFILING is enabled, but cProfile is not installed. No CPU profiling is happening.')
            return (None, None)
        try:
            import marshal
        except ImportError:
            logger.warning('RAY_SERVE_ENABLE_CPU_PROFILING is enabled, but marshal is not installed. No CPU profiling is happening.')
            return (None, None)
        logs_dir = get_serve_logs_dir()
        cpu_profiler_file_name = get_component_log_file_name(component_name=component_name, component_id=component_id, component_type=component_type, suffix='_cprofile.prof')
        cpu_profiler_file_path = os.path.join(logs_dir, cpu_profiler_file_name)
        profile = cProfile.Profile()
        profile.enable()
        logger.info('RAY_SERVE_ENABLE_CPU_PROFILING is enabled. Started cProfile on this actor.')
        return (profile, cpu_profiler_file_path)
    else:
        return (None, None)