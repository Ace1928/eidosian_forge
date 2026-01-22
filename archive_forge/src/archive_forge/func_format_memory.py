import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def format_memory(mem_bytes: Number) -> str:
    """Formats memory in bytes in friendly unit. E.g. (2**30 + 1) bytes should
    be displayed as 1GiB but 1 byte should be displayed as 1B, (as opposed to
    rounding it to 0GiB).
    """
    for suffix, bytes_per_unit in MEMORY_SUFFIXES:
        if mem_bytes >= bytes_per_unit:
            mem_in_unit = mem_bytes / bytes_per_unit
            return f'{mem_in_unit:.2f}{suffix}'
    return f'{int(mem_bytes)}B'