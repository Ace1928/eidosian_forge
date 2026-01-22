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
def should_update(self, node_id):
    if not self.can_update(node_id):
        return UpdateInstructions(None, None, None, None)
    status = self.provider.node_tags(node_id).get(TAG_RAY_NODE_STATUS)
    if status == STATUS_UP_TO_DATE and self.files_up_to_date(node_id):
        return UpdateInstructions(None, None, None, None)
    successful_updated = self.num_successful_updates.get(node_id, 0) > 0
    if successful_updated and self.config.get('restart_only', False):
        setup_commands = []
        ray_start_commands = self.config['worker_start_ray_commands']
    elif successful_updated and self.config.get('no_restart', False):
        setup_commands = self._get_node_type_specific_fields(node_id, 'worker_setup_commands')
        ray_start_commands = []
    else:
        setup_commands = self._get_node_type_specific_fields(node_id, 'worker_setup_commands')
        ray_start_commands = self.config['worker_start_ray_commands']
    docker_config = self._get_node_specific_docker_config(node_id)
    return UpdateInstructions(node_id=node_id, setup_commands=setup_commands, ray_start_commands=ray_start_commands, docker_config=docker_config)