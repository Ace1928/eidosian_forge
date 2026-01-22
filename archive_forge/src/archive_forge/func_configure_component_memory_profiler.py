import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def configure_component_memory_profiler(component_name: str, component_id: str, component_type: Optional[ServeComponentType]=None):
    """Configures the memory logger for this component.

    Does nothing if RAY_SERVE_ENABLE_MEMORY_PROFILING is disabled.
    """
    if RAY_SERVE_ENABLE_MEMORY_PROFILING:
        logger = logging.getLogger(SERVE_LOGGER_NAME)
        try:
            import memray
            logs_dir = get_serve_logs_dir()
            memray_file_name = get_component_log_file_name(component_name=component_name, component_id=component_id, component_type=component_type, suffix='_memray_0.bin')
            memray_file_path = os.path.join(logs_dir, memray_file_name)
            restart_counter = 1
            while os.path.exists(memray_file_path):
                memray_file_name = get_component_log_file_name(component_name=component_name, component_id=component_id, component_type=component_type, suffix=f'_memray_{restart_counter}.bin')
                memray_file_path = os.path.join(logs_dir, memray_file_name)
                restart_counter += 1
            memray.Tracker(memray_file_path, native_traces=True).__enter__()
            logger.info(f'RAY_SERVE_ENABLE_MEMORY_PROFILING is enabled. Started a memray tracker on this actor. Tracker file located at "{memray_file_path}"')
        except ImportError:
            logger.warning('RAY_SERVE_ENABLE_MEMORY_PROFILING is enabled, but memray is not installed. No memory profiling is happening. `pip install memray` to enable memory profiling.')