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
def generate_write_data(usage_stats: UsageStatsToReport, error: str) -> UsageStatsToWrite:
    """Generate the report data.

    Params:
        usage_stats: The usage stats that were reported.
        error: The error message of failed reports.

    Returns:
        UsageStatsToWrite
    """
    data = UsageStatsToWrite(usage_stats=usage_stats, success=error is None, error=error)
    return data