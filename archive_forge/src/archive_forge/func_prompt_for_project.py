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
def prompt_for_project(ctx, entity):
    """Ask the user for a project, creating one if necessary."""
    result = ctx.invoke(projects, entity=entity, display=False)
    api = _get_cling_api()
    try:
        if len(result) == 0:
            project = click.prompt('Enter a name for your first project')
            project = api.upsert_project(project, entity=entity)['name']
        else:
            project_names = [project['name'] for project in result] + ['Create New']
            wandb.termlog('Which project should we use?')
            result = util.prompt_choices(project_names)
            if result:
                project = result
            else:
                project = 'Create New'
            if project == 'Create New':
                project = click.prompt('Enter a name for your new project', value_proc=api.format_project)
                project = api.upsert_project(project, entity=entity)['name']
    except wandb.errors.CommError as e:
        raise ClickException(str(e))
    return project