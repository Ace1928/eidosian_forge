import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def log_access_log_filter(record: logging.LogRecord) -> bool:
    """Filters ray serve access log based on 'serve_access_log' key in `extra` dict."""
    if not hasattr(record, 'serve_access_log') or record.serve_access_log is None:
        return True
    return not record.serve_access_log