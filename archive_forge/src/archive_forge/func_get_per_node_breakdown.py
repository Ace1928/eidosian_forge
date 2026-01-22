import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def get_per_node_breakdown(lm_summary: LoadMetricsSummary, node_type_mapping: Optional[Dict[str, float]], node_activities: Optional[Dict[str, List[str]]], verbose: bool) -> str:
    sio = StringIO()
    if node_type_mapping is None:
        node_type_mapping = {}
    print(file=sio)
    for node_id, usage in lm_summary.usage_by_node.items():
        print(file=sio)
        node_string = f'Node: {node_id}'
        if node_id in node_type_mapping:
            node_type = node_type_mapping[node_id]
            node_string += f' ({node_type})'
        print(node_string, file=sio)
        if lm_summary.idle_time_map and node_id in lm_summary.idle_time_map and (lm_summary.idle_time_map[node_id] > 0):
            print(f' Idle: {lm_summary.idle_time_map[node_id]} ms', file=sio)
        print(' Usage:', file=sio)
        for line in parse_usage(usage, verbose):
            print(f'  {line}', file=sio)
        if not node_activities:
            continue
        print(' Activity:', file=sio)
        if node_id not in node_activities:
            print('  (no activity)', file=sio)
        else:
            _, reasons = node_activities[node_id]
            for reason in reasons:
                print(f'  {reason}', file=sio)
    return sio.getvalue()