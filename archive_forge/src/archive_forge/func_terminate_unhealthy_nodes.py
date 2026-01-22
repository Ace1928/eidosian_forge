import copy
import logging
import math
import operator
import os
import queue
import subprocess
import threading
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import yaml
import ray
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.legacy_info_string import legacy_log_info_string
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.local.node_provider import (
from ray.autoscaler._private.node_launcher import BaseNodeLauncher, NodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.node_tracker import NodeTracker
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler._private.resource_demand_scheduler import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.exceptions import RpcError
def terminate_unhealthy_nodes(self, now: float):
    """Terminated nodes for which we haven't received a heartbeat on time.
        These nodes are subsequently terminated.
        """
    assert self.provider
    assert self.non_terminated_nodes
    for node_id in self.non_terminated_nodes.worker_ids:
        node_status = self.provider.node_tags(node_id)[TAG_RAY_NODE_STATUS]
        if not node_status == STATUS_UP_TO_DATE:
            continue
        ip = self.provider.internal_ip(node_id)
        if ip not in self.load_metrics.last_heartbeat_time_by_ip:
            self.load_metrics.mark_active(ip)
        if self.heartbeat_on_time(node_id, now):
            continue
        self.schedule_node_termination(node_id, 'lost contact with raylet', logger.warning)
    self.terminate_scheduled_nodes()