import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def fetch_wandb_project_run_info(entity: str, project: str, run_name: str, api: Api) -> Any:
    _logger.info('Fetching run info...')
    try:
        result = api.get_run_info(entity, project, run_name)
    except CommError:
        result = None
    if result is None:
        raise LaunchError(f"Run info is invalid or doesn't exist for {api.settings('base_url')}/{entity}/{project}/runs/{run_name}")
    if result.get('codePath') is None:
        metadata = api.download_url(project, 'wandb-metadata.json', run=run_name, entity=entity)
        if metadata is not None:
            _, response = api.download_file(metadata['url'])
            data = response.json()
            result['codePath'] = data.get('codePath')
            result['cudaVersion'] = data.get('cuda', None)
    return result