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
def emit_metrics(self, load_metrics_summary, autoscaler_summary, node_types):
    if autoscaler_summary is None:
        return None
    for resource_name in ['CPU', 'GPU', 'TPU']:
        _, total = load_metrics_summary.usage.get(resource_name, (0, 0))
        pending = autoscaler_summary.pending_resources.get(resource_name, 0)
        self.prom_metrics.cluster_resources.labels(resource=resource_name, SessionName=self.prom_metrics.session_name).set(total)
        self.prom_metrics.pending_resources.labels(resource=resource_name, SessionName=self.prom_metrics.session_name).set(pending)
    pending_node_count = Counter()
    for _, node_type, _ in autoscaler_summary.pending_nodes:
        pending_node_count[node_type] += 1
    for node_type, count in autoscaler_summary.pending_launches.items():
        pending_node_count[node_type] += count
    for node_type in node_types:
        count = pending_node_count[node_type]
        self.prom_metrics.pending_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)
    for node_type in node_types:
        count = autoscaler_summary.active_nodes.get(node_type, 0)
        self.prom_metrics.active_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)
    failed_node_counts = Counter()
    for _, node_type in autoscaler_summary.failed_nodes:
        failed_node_counts[node_type] += 1
    for node_type, count in failed_node_counts.items():
        self.prom_metrics.recently_failed_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)