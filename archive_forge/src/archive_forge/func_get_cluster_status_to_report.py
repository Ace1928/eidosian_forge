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
def get_cluster_status_to_report(gcs_client) -> ClusterStatusToReport:
    """Get the current status of this cluster.

    It is a blocking API.

    Params:
        gcs_client: The GCS client to perform KV operation GET.

    Returns:
        The current cluster status or empty if it fails to get that information.
    """
    try:
        from ray.autoscaler.v2.utils import is_autoscaler_v2
        if is_autoscaler_v2():
            return _get_cluster_status_to_report_v2(gcs_client)
        cluster_status = gcs_client.internal_kv_get(ray._private.ray_constants.DEBUG_AUTOSCALING_STATUS.encode(), namespace=None)
        if not cluster_status:
            return ClusterStatusToReport()
        result = ClusterStatusToReport()
        to_GiB = 1 / 2 ** 30
        cluster_status = json.loads(cluster_status.decode('utf-8'))
        if 'load_metrics_report' not in cluster_status or 'usage' not in cluster_status['load_metrics_report']:
            return ClusterStatusToReport()
        usage = cluster_status['load_metrics_report']['usage']
        if 'CPU' in usage:
            result.total_num_cpus = int(usage['CPU'][1])
        if 'GPU' in usage:
            result.total_num_gpus = int(usage['GPU'][1])
        if 'memory' in usage:
            result.total_memory_gb = usage['memory'][1] * to_GiB
        if 'object_store_memory' in usage:
            result.total_object_store_memory_gb = usage['object_store_memory'][1] * to_GiB
        return result
    except Exception as e:
        logger.info(f'Failed to get cluster status to report {e}')
        return ClusterStatusToReport()