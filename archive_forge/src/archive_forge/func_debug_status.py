import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def debug_status(status, error, verbose: bool=False, address: Optional[str]=None) -> str:
    """
    Return a debug string for the autoscaler.

    Args:
        status: The autoscaler status string for v1
        error: The autoscaler error string for v1
        verbose: Whether to print verbose information.
        address: The address of the cluster (gcs address).

    Returns:
        str: A debug string for the cluster's status.
    """
    from ray.autoscaler.v2.utils import is_autoscaler_v2
    if is_autoscaler_v2():
        from ray.autoscaler.v2.sdk import get_cluster_status
        from ray.autoscaler.v2.utils import ClusterStatusFormatter
        cluster_status = get_cluster_status(address)
        status = ClusterStatusFormatter.format(cluster_status, verbose=verbose)
    elif status:
        status = status.decode('utf-8')
        status_dict = json.loads(status)
        lm_summary_dict = status_dict.get('load_metrics_report')
        autoscaler_summary_dict = status_dict.get('autoscaler_report')
        timestamp = status_dict.get('time')
        gcs_request_time = status_dict.get('gcs_request_time')
        non_terminated_nodes_time = status_dict.get('non_terminated_nodes_time')
        if lm_summary_dict and autoscaler_summary_dict and timestamp:
            lm_summary = LoadMetricsSummary(**lm_summary_dict)
            node_availability_summary_dict = autoscaler_summary_dict.pop('node_availability_summary', {})
            node_availability_summary = NodeAvailabilitySummary.from_fields(**node_availability_summary_dict)
            autoscaler_summary = AutoscalerSummary(node_availability_summary=node_availability_summary, **autoscaler_summary_dict)
            report_time = datetime.datetime.fromtimestamp(timestamp)
            status = format_info_string(lm_summary, autoscaler_summary, time=report_time, gcs_request_time=gcs_request_time, non_terminated_nodes_time=non_terminated_nodes_time, verbose=verbose)
        else:
            status = 'No cluster status. It may take a few seconds for the Ray internal services to start up.'
    else:
        status = 'No cluster status. It may take a few seconds for the Ray internal services to start up.'
    if error:
        status += '\n'
        status += error.decode('utf-8')
    return status