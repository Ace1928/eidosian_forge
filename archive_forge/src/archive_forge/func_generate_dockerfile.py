import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def generate_dockerfile(launch_project: LaunchProject, entry_point: EntryPoint, runner_type: str, builder_type: str, dockerfile: Optional[str]=None) -> str:
    if launch_project.project_dir is not None and dockerfile:
        path = os.path.join(launch_project.project_dir, dockerfile)
        if not os.path.exists(path):
            raise LaunchError(f'Dockerfile does not exist at {path}')
        launch_project.project_dir = os.path.dirname(path)
        wandb.termlog(f'Using dockerfile: {dockerfile}')
        return open(path).read()
    if launch_project.python_version:
        spl = launch_project.python_version.split('.')[:2]
        py_version, py_major = ('.'.join(spl), spl[0])
    else:
        py_version, py_major = get_current_python_version()
    if launch_project.deps_type == 'pip' or launch_project.deps_type is None:
        python_build_image = f'python:{py_version}'
    elif launch_project.deps_type == 'conda':
        python_build_image = 'continuumio/miniconda3:latest' if py_major == '3' else 'continuumio/miniconda:latest'
    requirements_section = get_requirements_section(launch_project, builder_type)
    python_base_setup = get_base_setup(launch_project, py_version, py_major)
    username, userid = get_docker_user(launch_project, runner_type)
    user_setup = get_user_setup(username, userid, runner_type)
    workdir = f'/home/{username}'
    entrypoint_section = get_entrypoint_setup(entry_point)
    dockerfile_contents = DOCKERFILE_TEMPLATE.format(py_build_image=python_build_image, requirements_section=requirements_section, base_setup=python_base_setup, uid=userid, user_setup=user_setup, workdir=workdir, entrypoint_section=entrypoint_section)
    return dockerfile_contents