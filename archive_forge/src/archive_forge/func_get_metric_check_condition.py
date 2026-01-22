import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_metric_check_condition(metrics_to_check: List[MetricSamplePattern], export_addr: Optional[str]=None) -> Callable[[], bool]:
    """A condition to check if a prometheus metrics reach a certain value.
    This is a blocking check that can be passed into a `wait_for_condition`
    style function.

    Args:
      metrics_to_check: A list of MetricSamplePattern. The fields that
      aren't `None` will be matched.

    Returns:
      A function that returns True if all the metrics are emitted.

    """
    node_info = ray.nodes()[0]
    metrics_export_port = node_info['MetricsExportPort']
    addr = node_info['NodeManagerAddress']
    prom_addr = export_addr or f'{addr}:{metrics_export_port}'

    def f():
        for metric_pattern in metrics_to_check:
            _, metric_names, metric_samples = fetch_prometheus([prom_addr])
            for metric_sample in metric_samples:
                if metric_pattern.matches(metric_sample):
                    break
            else:
                print(f"Didn't find {metric_pattern}", 'all samples', metric_samples)
                return False
        return True
    return f