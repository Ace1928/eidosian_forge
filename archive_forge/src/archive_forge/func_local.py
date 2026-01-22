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
@cli.command(context_settings=RUN_CONTEXT, help='Start a local W&B container (deprecated, see wandb server --help)', hidden=True)
@click.pass_context
@click.option('--port', '-p', default='8080', help='The host port to bind W&B local on')
@click.option('--env', '-e', default=[], multiple=True, help='Env vars to pass to wandb/local')
@click.option('--daemon/--no-daemon', default=True, help="Run or don't run in daemon mode")
@click.option('--upgrade', is_flag=True, default=False, help='Upgrade to the most recent version')
@click.option('--edge', is_flag=True, default=False, help='Run the bleeding edge', hidden=True)
@display_error
def local(ctx, *args, **kwargs):
    wandb.termwarn('`wandb local` has been replaced with `wandb server start`.')
    ctx.invoke(start, *args, **kwargs)