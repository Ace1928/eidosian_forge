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
def spawn_updater(self, node_id, setup_commands, ray_start_commands, node_resources, node_labels, docker_config):
    logger.info(f'Creating new (spawn_updater) updater thread for node {node_id}.')
    ip = self.provider.internal_ip(node_id)
    node_type = self._get_node_type(node_id)
    self.node_tracker.track(node_id, ip, node_type)
    head_node_ip = self.provider.internal_ip(self.non_terminated_nodes.head_id)
    updater = NodeUpdaterThread(node_id=node_id, provider_config=self.config['provider'], provider=self.provider, auth_config=self.config['auth'], cluster_name=self.config['cluster_name'], file_mounts=self.config['file_mounts'], initialization_commands=with_head_node_ip(self._get_node_type_specific_fields(node_id, 'initialization_commands'), head_node_ip), setup_commands=with_head_node_ip(setup_commands, head_node_ip), ray_start_commands=with_head_node_ip(ray_start_commands, head_node_ip), runtime_hash=self.runtime_hash, file_mounts_contents_hash=self.file_mounts_contents_hash, is_head_node=False, cluster_synced_files=self.config['cluster_synced_files'], rsync_options={'rsync_exclude': self.config.get('rsync_exclude'), 'rsync_filter': self.config.get('rsync_filter')}, process_runner=self.process_runner, use_internal_ip=True, docker_config=docker_config, node_resources=node_resources, node_labels=node_labels)
    updater.start()
    self.updaters[node_id] = updater