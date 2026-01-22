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
def is_bare_wandb_uri(uri: str) -> bool:
    """Check that a wandb uri is valid.

    URI must be in the format
    `/<entity>/<project>/runs/<run_name>[other stuff]`
    or
    `/<entity>/<project>/artifacts/job/<job_name>[other stuff]`.
    """
    _logger.info(f'Checking if uri {uri} is bare...')
    return uri.startswith('/') and WandbReference.is_uri_job_or_run(uri)