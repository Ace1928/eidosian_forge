import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def get_component_log_file_name(component_name: str, component_id: str, component_type: Optional[ServeComponentType], suffix: str='') -> str:
    """Get the component's log file name."""
    component_log_file_name = component_name
    if component_type is not None:
        component_log_file_name = f'{component_type}_{component_name}'
        if component_type != ServeComponentType.REPLICA:
            component_name = f'{component_type}_{component_name}'
    log_file_name = LOG_FILE_FMT.format(component_name=component_log_file_name, component_id=component_id, suffix=suffix)
    return log_file_name