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
@cli.command(hidden=True)
@click.argument('cluster_config_file', required=True, type=str)
@click.option('--yes', '-y', is_flag=True, default=False, help="Don't ask for confirmation.")
@click.option('--hard', is_flag=True, default=False, help="Terminates the node via node provider (defaults to a 'soft kill' which terminates Ray but does not actually delete the instances).")
@click.option('--cluster-name', '-n', required=False, type=str, help='Override the configured cluster name.')
def kill_random_node(cluster_config_file, yes, hard, cluster_name):
    """Kills a random Ray node. For testing purposes only."""
    click.echo('Killed node with IP ' + kill_node(cluster_config_file, yes, hard, cluster_name))