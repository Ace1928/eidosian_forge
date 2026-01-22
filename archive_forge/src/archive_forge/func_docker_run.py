import asyncio
import configparser
import datetime
import getpass
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from functools import wraps
from typing import Any, Dict, Optional
import click
import yaml
from click.exceptions import ClickException
from dockerpycreds.utils import find_executable
import wandb
import wandb.env
import wandb.sdk.verify.verify as wandb_verify
from wandb import Config, Error, env, util, wandb_agent, wandb_sdk
from wandb.apis import InternalApi, PublicApi
from wandb.apis.public import RunQueue
from wandb.integration.magic import magic_install
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.launch import utils as launch_utils
from wandb.sdk.launch._launch_add import _launch_add
from wandb.sdk.launch.errors import ExecutionError, LaunchError
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.wburls import wburls
from wandb.sync import SyncManager, get_run_from_path, get_runs
import __main__
@cli.command(context_settings=RUN_CONTEXT, name='docker-run')
@click.pass_context
@click.argument('docker_run_args', nargs=-1)
def docker_run(ctx, docker_run_args):
    """Wrap `docker run` and adds WANDB_API_KEY and WANDB_DOCKER environment variables.

    This will also set the runtime to nvidia if the nvidia-docker executable is present
    on the system and --runtime wasn't set.

    See `docker run --help` for more details.
    """
    api = InternalApi()
    args = list(docker_run_args)
    if len(args) > 0 and args[0] == 'run':
        args.pop(0)
    if len([a for a in args if a.startswith('--runtime')]) == 0 and find_executable('nvidia-docker'):
        args = ['--runtime', 'nvidia'] + args
    image = util.image_from_docker_args(args)
    resolved_image = None
    if image:
        resolved_image = wandb.docker.image_id(image)
    if resolved_image:
        args = ['-e', 'WANDB_DOCKER=%s' % resolved_image] + args
    else:
        wandb.termlog("Couldn't detect image argument, running command without the WANDB_DOCKER env variable")
    if api.api_key:
        args = ['-e', 'WANDB_API_KEY=%s' % api.api_key] + args
    else:
        wandb.termlog('Not logged in, run `wandb login` from the host machine to enable result logging')
    subprocess.call(['docker', 'run'] + args)