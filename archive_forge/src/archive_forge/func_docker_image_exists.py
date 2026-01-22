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
def docker_image_exists(docker_image: str, should_raise: bool=False) -> bool:
    """Check if a specific image is already available.

    Optionally raises an exception if the image is not found.
    """
    _logger.info('Checking if base image exists...')
    try:
        docker.run(['docker', 'image', 'inspect', docker_image])
        return True
    except (docker.DockerError, ValueError) as e:
        if should_raise:
            raise e
        _logger.info('Base image not found. Generating new base image')
        return False