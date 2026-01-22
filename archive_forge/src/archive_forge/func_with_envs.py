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
def with_envs(cmds: List[str], kv: Dict[str, str]) -> str:
    """
    Returns a list of commands with the given environment variables set.

    Args:
        cmds (List[str]): List of commands to set environment variables for.
        kv (Dict[str, str]): Dictionary of environment variables to set.

    Returns:
        List[str]: List of commands with the given environment variables set.

    Example:
        with_envs(["echo $FOO"], {"FOO": "BAR"})
            -> ["export FOO=BAR; echo $FOO"]
    """
    out_cmds = []
    for cmd in cmds:
        kv_str = ''
        for k, v in kv.items():
            kv_str += f'export {k}={v}; '
        out_cmds.append(f'{kv_str}{cmd}')
    return out_cmds