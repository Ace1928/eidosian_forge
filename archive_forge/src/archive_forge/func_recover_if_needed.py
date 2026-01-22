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
def recover_if_needed(self, node_id, now):
    if not self.can_update(node_id):
        return
    if self.heartbeat_on_time(node_id, now):
        return
    logger.warning('StandardAutoscaler: {}: No recent heartbeat, restarting Ray to recover...'.format(node_id))
    self.event_summarizer.add('Restarting {} nodes of type ' + self._get_node_type(node_id) + ' (lost contact with raylet).', quantity=1, aggregate=operator.add)
    head_node_ip = self.provider.internal_ip(self.non_terminated_nodes.head_id)
    updater = NodeUpdaterThread(node_id=node_id, provider_config=self.config['provider'], provider=self.provider, auth_config=self.config['auth'], cluster_name=self.config['cluster_name'], file_mounts={}, initialization_commands=[], setup_commands=[], ray_start_commands=with_head_node_ip(self.config['worker_start_ray_commands'], head_node_ip), runtime_hash=self.runtime_hash, file_mounts_contents_hash=self.file_mounts_contents_hash, process_runner=self.process_runner, use_internal_ip=True, is_head_node=False, docker_config=self.config.get('docker'), node_resources=self._node_resources(node_id), node_labels=self._node_labels(node_id), for_recovery=True)
    updater.start()
    self.updaters[node_id] = updater