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
def teardown_cluster(config_file: str, yes: bool, workers_only: bool, override_cluster_name: Optional[str], keep_min_workers: bool) -> None:
    """Destroys all nodes of a Ray cluster described by a config json."""
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config['cluster_name'] = override_cluster_name
    config = _bootstrap_config(config)
    cli_logger.confirm(yes, 'Destroying cluster.', _abort=True)
    if not workers_only:
        try:
            exec_cluster(config_file, cmd='ray stop', run_env='auto', screen=False, tmux=False, stop=False, start=False, override_cluster_name=override_cluster_name, port_forward=None, with_output=False)
        except Exception as e:
            cli_logger.verbose_error('{}', str(e))
            cli_logger.warning('Exception occurred when stopping the cluster Ray runtime (use -v to dump teardown exceptions).')
            cli_logger.warning('Ignoring the exception and attempting to shut down the cluster nodes anyway.')
    provider = _get_node_provider(config['provider'], config['cluster_name'])

    def remaining_nodes():
        workers = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
        if keep_min_workers:
            min_workers = config.get('min_workers', 0)
            cli_logger.print('{} random worker nodes will not be shut down. ' + cf.dimmed('(due to {})'), cf.bold(min_workers), cf.bold('--keep-min-workers'))
            workers = random.sample(workers, len(workers) - min_workers)
        if workers_only:
            cli_logger.print('The head node will not be shut down. ' + cf.dimmed('(due to {})'), cf.bold('--workers-only'))
            return workers
        head = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_HEAD})
        return head + workers

    def run_docker_stop(node, container_name):
        try:
            updater = NodeUpdaterThread(node_id=node, provider_config=config['provider'], provider=provider, auth_config=config['auth'], cluster_name=config['cluster_name'], file_mounts=config['file_mounts'], initialization_commands=[], setup_commands=[], ray_start_commands=[], runtime_hash='', file_mounts_contents_hash='', is_head_node=False, docker_config=config.get('docker'))
            _exec(updater, f'docker stop {container_name}', with_output=False, run_env='host')
        except Exception:
            cli_logger.warning(f'Docker stop failed on {node}')
    A = remaining_nodes()
    container_name = config.get('docker', {}).get('container_name')
    if container_name:
        output_redir = cmd_output_util.is_output_redirected()
        cmd_output_util.set_output_redirected(True)
        allow_interactive = cmd_output_util.does_allow_interactive()
        cmd_output_util.set_allow_interactive(False)
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SHUTDOWN_WORKERS) as executor:
            for node in A:
                executor.submit(run_docker_stop, node=node, container_name=container_name)
        cmd_output_util.set_output_redirected(output_redir)
        cmd_output_util.set_allow_interactive(allow_interactive)
    with LogTimer('teardown_cluster: done.'):
        while A:
            provider.terminate_nodes(A)
            cli_logger.print('Requested {} nodes to shut down.', cf.bold(len(A)), _tags=dict(interval='1s'))
            time.sleep(POLL_INTERVAL)
            A = remaining_nodes()
            cli_logger.print('{} nodes remaining after {} second(s).', cf.bold(len(A)), POLL_INTERVAL)
        cli_logger.success('No nodes remaining.')