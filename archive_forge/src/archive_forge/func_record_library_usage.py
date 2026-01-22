import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def record_library_usage(library_usage: str):
    """Record library usage (e.g. which library is used)"""
    with _recorded_library_usages_lock:
        if library_usage in _recorded_library_usages:
            return
        _recorded_library_usages.add(library_usage)
    if not _internal_kv_initialized():
        return
    if ray._private.worker.global_worker.mode == ray.SCRIPT_MODE or ray._private.worker.global_worker.mode == ray.WORKER_MODE or ray.util.client.ray.is_connected():
        _put_library_usage(library_usage)