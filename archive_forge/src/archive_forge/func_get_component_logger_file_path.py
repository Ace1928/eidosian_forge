import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def get_component_logger_file_path() -> Optional[str]:
    """Returns the relative file path for the Serve logger, if it exists.

    If a logger was configured through configure_component_logger() for the Serve
    component that's calling this function, this returns the location of the log file
    relative to the ray logs directory.
    """
    logger = logging.getLogger(SERVE_LOGGER_NAME)
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            absolute_path = handler.baseFilename
            ray_logs_dir = ray._private.worker._global_node.get_logs_dir_path()
            if absolute_path.startswith(ray_logs_dir):
                return absolute_path[len(ray_logs_dir):]