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
def write_usage_data(self, data: UsageStatsToWrite, dir_path: str) -> None:
    """Write the usage data to the directory.

        Params:
            data: Data to report
            dir_path: The path to the directory to write usage data.
        """
    dir_path = Path(dir_path)
    destination = dir_path / usage_constant.USAGE_STATS_FILE
    temp = dir_path / f'{usage_constant.USAGE_STATS_FILE}.tmp'
    with temp.open(mode='w') as json_file:
        json_file.write(json.dumps(asdict(data)))
    if sys.platform == 'win32':
        destination.unlink(missing_ok=True)
    temp.rename(destination)