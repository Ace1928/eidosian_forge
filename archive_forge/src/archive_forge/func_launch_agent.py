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
@cli.command(context_settings=CONTEXT, help='Run a W&B launch agent.')
@click.pass_context
@click.option('--queue', '-q', 'queues', default=None, multiple=True, metavar='<queue(s)>', help='The name of a queue for the agent to watch. Multiple -q flags supported.')
@click.option('--project', '-p', default=None, help='Name of the project which the agent will watch.\n    If passed in, will override the project value passed in using a config file.')
@click.option('--entity', '-e', default=None, help='The entity to use. Defaults to current logged-in user')
@click.option('--log-file', '-l', default=None, help='Destination for internal agent logs. Use - for stdout. By default all agents logs will go to debug.log in your wandb/ subdirectory or WANDB_DIR if set.')
@click.option('--max-jobs', '-j', default=None, help='The maximum number of launch jobs this agent can run in parallel. Defaults to 1. Set to -1 for no upper limit')
@click.option('--config', '-c', default=None, help='path to the agent config yaml to use')
@click.option('--url', '-u', default=None, hidden=True, help='a wandb client registration URL, this is generated in the UI')
@display_error
def launch_agent(ctx, project=None, entity=None, queues=None, max_jobs=None, config=None, url=None, log_file=None):
    logger.info(f'=== Launch-agent called with kwargs {locals()}  CLI Version: {wandb.__version__} ===')
    if url is not None:
        raise LaunchError('--url is not supported in this version, upgrade with: pip install -u wandb')
    import wandb.sdk.launch._launch as _launch
    if log_file is not None:
        _launch.set_launch_logfile(log_file)
    api = _get_cling_api()
    wandb._sentry.configure_scope(process_context='launch_agent')
    agent_config, api = _launch.resolve_agent_config(entity, project, max_jobs, queues, config)
    if len(agent_config.get('queues')) == 0:
        raise LaunchError('To launch an agent please specify a queue or a list of queues in the configuration file or cli.')
    launch_utils.check_logged_in(api)
    wandb.termlog('Starting launch agent ✨')
    try:
        _launch.create_and_run_agent(api, agent_config)
    except Exception as e:
        wandb._sentry.exception(e)
        raise e