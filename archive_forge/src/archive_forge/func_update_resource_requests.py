import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union
import ray
import ray._private.ray_constants as ray_constants
import ray._private.utils
from ray._private.event.event_logger import get_event_logger
from ray._private.ray_logging import setup_component_logger
from ray._raylet import GcsClient
from ray.autoscaler._private.autoscaler import StandardAutoscaler
from ray.autoscaler._private.commands import teardown_cluster
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import format_readonly_node_type
from ray.core.generated import gcs_pb2
from ray.core.generated.event_pb2 import Event as RayEvent
from ray.experimental.internal_kv import (
def update_resource_requests(self):
    """Fetches resource requests from the internal KV and updates load."""
    if not _internal_kv_initialized():
        return
    data = _internal_kv_get(ray._private.ray_constants.AUTOSCALER_RESOURCE_REQUEST_CHANNEL)
    if data:
        try:
            resource_request = json.loads(data)
            self.load_metrics.set_resource_requests(resource_request)
        except Exception:
            logger.exception('Error parsing resource requests')