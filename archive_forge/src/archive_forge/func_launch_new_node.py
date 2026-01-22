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
def launch_new_node(self, count: int, node_type: str) -> None:
    logger.info('StandardAutoscaler: Queue {} new nodes for launch'.format(count))
    self.pending_launches.inc(node_type, count)
    config = copy.deepcopy(self.config)
    if self.foreground_node_launch:
        assert self.foreground_node_launcher is not None
        self.foreground_node_launcher.launch_node(config, count, node_type)
    else:
        assert self.launch_queue is not None
        while count > 0:
            self.launch_queue.put((config, min(count, self.max_launch_batch), node_type))
            count -= self.max_launch_batch