import json
import logging
import os
import re
import threading
import time
import traceback
from collections import namedtuple
from typing import List, Tuple, Any, Dict
from prometheus_client.core import (
from opencensus.metrics.export.value import ValueDouble
from opencensus.stats import aggregation
from opencensus.stats import measure as measure_module
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats.base_exporter import StatsExporter
from prometheus_client.core import Metric as PrometheusMetric
from opencensus.stats.aggregation_data import (
from opencensus.stats.view import View
from opencensus.tags import tag_key as tag_key_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.tags import tag_value as tag_value_module
import ray
from ray._raylet import GcsClient
from ray.core.generated.metrics_pb2 import Metric
def get_file_discovery_content(self):
    """Return the content for Prometheus service discovery."""
    nodes = ray.nodes()
    metrics_export_addresses = ['{}:{}'.format(node['NodeManagerAddress'], node['MetricsExportPort']) for node in nodes if node['alive'] is True]
    gcs_client = GcsClient(address=self.gcs_address)
    autoscaler_addr = gcs_client.internal_kv_get(b'AutoscalerMetricsAddress', None)
    if autoscaler_addr:
        metrics_export_addresses.append(autoscaler_addr.decode('utf-8'))
    dashboard_addr = gcs_client.internal_kv_get(b'DashboardMetricsAddress', None)
    if dashboard_addr:
        metrics_export_addresses.append(dashboard_addr.decode('utf-8'))
    return json.dumps([{'labels': {'job': 'ray'}, 'targets': metrics_export_addresses}])