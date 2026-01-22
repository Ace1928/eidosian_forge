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
def get_local_dump_archive(stream: bool=False, output: Optional[str]=None, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=False, tempfile: Optional[str]=None) -> Optional[str]:
    if stream and output:
        raise ValueError('You can only use either `--output` or `--stream`, but not both.')
    parameters = GetParameters(logs=logs, debug_state=debug_state, pip=pip, processes=processes, processes_verbose=processes_verbose)
    with Archive(file=tempfile) as archive:
        get_all_local_data(archive, parameters)
    tmp = archive.file
    if stream:
        with open(tmp, 'rb') as fp:
            os.write(1, fp.read())
        os.remove(tmp)
        return None
    target = output or os.path.join(os.getcwd(), os.path.basename(tmp))
    shutil.move(tmp, target)
    cli_logger.print(f'Created local data archive at {target}')
    return target