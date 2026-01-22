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
def fetch_prometheus_metrics(prom_addresses: List[str]) -> Dict[str, List[Any]]:
    """Return prometheus metrics from the given addresses.

    Args:
        prom_addresses: List of metrics_agent addresses to collect metrics from.

    Returns:
        Dict mapping from metric name to list of samples for the metric.
    """
    _, _, samples = fetch_prometheus(prom_addresses)
    samples_by_name = defaultdict(list)
    for sample in samples:
        samples_by_name[sample.name].append(sample)
    return samples_by_name