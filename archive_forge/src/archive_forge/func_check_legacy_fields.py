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
def check_legacy_fields(config: Dict[str, Any]) -> None:
    """For use in providers that have completed the migration to
    available_node_types.

    Warns user that head_node and worker_nodes fields are being ignored.
    Throws an error if available_node_types and head_node_type aren't
    specified.
    """
    if 'head_node' in config and config['head_node']:
        cli_logger.warning('The `head_node` field is deprecated and will be ignored. Use `head_node_type` and `available_node_types` instead.')
    if 'worker_nodes' in config and config['worker_nodes']:
        cli_logger.warning('The `worker_nodes` field is deprecated and will be ignored. Use `available_node_types` instead.')
    if 'available_node_types' not in config:
        cli_logger.error('`available_node_types` not specified in config')
        raise ValueError('`available_node_types` not specified in config')
    if 'head_node_type' not in config:
        cli_logger.error('`head_node_type` not specified in config')
        raise ValueError('`head_node_type` not specified in config')