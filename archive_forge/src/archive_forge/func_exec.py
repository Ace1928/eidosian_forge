import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib
import urllib.parse
import warnings
import shutil
from datetime import datetime
from typing import Optional, Set, List, Tuple
import click
import psutil
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._private.utils import (
from ray._private.internal_api import memory_summary
from ray._private.storage import _load_class
from ray._private.usage import usage_lib
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.autoscaler._private.commands import (
from ray.autoscaler._private.constants import RAY_PROCESSES
from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID
from ray.util.annotations import PublicAPI
@cli.command()
@click.argument('cluster_config_file', required=True, type=str)
@click.argument('cmd', required=True, type=str)
@click.option('--run-env', required=False, type=click.Choice(RUN_ENV_TYPES), default='auto', help='Choose whether to execute this command in a container or directly on the cluster head. Only applies when docker is configured in the YAML.')
@click.option('--stop', is_flag=True, default=False, help='Stop the cluster after the command finishes running.')
@click.option('--start', is_flag=True, default=False, help='Start the cluster if needed.')
@click.option('--screen', is_flag=True, default=False, help='Run the command in a screen.')
@click.option('--tmux', is_flag=True, default=False, help='Run the command in tmux.')
@click.option('--cluster-name', '-n', required=False, type=str, help='Override the configured cluster name.')
@click.option('--no-config-cache', is_flag=True, default=False, help='Disable the local cluster config cache.')
@click.option('--port-forward', '-p', required=False, multiple=True, type=int, help='Port to forward. Use this multiple times to forward multiple ports.')
@click.option('--disable-usage-stats', is_flag=True, default=False, help='If True, the usage stats collection will be disabled.')
@add_click_logging_options
def exec(cluster_config_file, cmd, run_env, screen, tmux, stop, start, cluster_name, no_config_cache, port_forward, disable_usage_stats):
    """Execute a command via SSH on a Ray cluster."""
    port_forward = [(port, port) for port in list(port_forward)]
    if start:
        if disable_usage_stats:
            usage_lib.set_usage_stats_enabled_via_env_var(False)
    exec_cluster(cluster_config_file, cmd=cmd, run_env=run_env, screen=screen, tmux=tmux, stop=stop, start=start, override_cluster_name=cluster_name, no_config_cache=no_config_cache, port_forward=port_forward, _allow_uninitialized_state=True)