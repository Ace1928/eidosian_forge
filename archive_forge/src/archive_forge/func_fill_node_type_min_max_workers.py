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
def fill_node_type_min_max_workers(config):
    """Sets default per-node max workers to global max_workers.
    This equivalent to setting the default per-node max workers to infinity,
    with the only upper constraint coming from the global max_workers.
    Sets default per-node min workers to zero.
    Also sets default max_workers for the head node to zero.
    """
    assert 'max_workers' in config, 'Global max workers should be set.'
    node_types = config['available_node_types']
    for node_type_name in node_types:
        node_type_data = node_types[node_type_name]
        node_type_data.setdefault('min_workers', 0)
        if 'max_workers' not in node_type_data:
            if node_type_name == config['head_node_type']:
                logger.info('setting max workers for head node type to 0')
                node_type_data.setdefault('max_workers', 0)
            else:
                global_max_workers = config['max_workers']
                logger.info(f'setting max workers for {node_type_name} to {global_max_workers}')
                node_type_data.setdefault('max_workers', global_max_workers)