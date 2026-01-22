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
@click.option('-v', '--verbose', is_flag=True)
@click.option('--dryrun', is_flag=True, help='Identifies the wheel but does not execute the installation.')
def install_nightly(verbose, dryrun):
    """Install the latest wheels for Ray.

    This uses the same python environment as the one that Ray is currently
    installed in. Make sure that there is no Ray processes on this
    machine (ray stop) when running this command.
    """
    raydir = os.path.abspath(os.path.dirname(ray.__file__))
    all_wheels_path = os.path.join(raydir, 'nightly-wheels.yaml')
    wheels = None
    if os.path.exists(all_wheels_path):
        with open(all_wheels_path) as f:
            wheels = yaml.safe_load(f)
    if not wheels:
        raise click.ClickException(f"Wheels not found in '{all_wheels_path}'! Please visit https://docs.ray.io/en/master/installation.html to obtain the latest wheels.")
    platform = sys.platform
    py_version = '{0}.{1}'.format(*sys.version_info[:2])
    matching_wheel = None
    for target_platform, wheel_map in wheels.items():
        if verbose:
            print(f'Evaluating os={target_platform}, python={list(wheel_map)}')
        if platform.startswith(target_platform):
            if py_version in wheel_map:
                matching_wheel = wheel_map[py_version]
                break
        if verbose:
            print('Not matched.')
    if matching_wheel is None:
        raise click.ClickException('Unable to identify a matching platform. Please visit https://docs.ray.io/en/master/installation.html to obtain the latest wheels.')
    if dryrun:
        print(f'Found wheel: {matching_wheel}')
    else:
        cmd = [sys.executable, '-m', 'pip', 'install', '-U', matching_wheel]
        print(f'Running: {' '.join(cmd)}.')
        subprocess.check_call(cmd)