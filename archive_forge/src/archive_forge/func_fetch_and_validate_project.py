import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def fetch_and_validate_project(self) -> None:
    """Fetches a project into a local directory, adds the config values to the directory, and validates the first entrypoint for the project.

        Arguments:
            launch_project: LaunchProject to fetch and validate.
            api: Instance of wandb.apis.internal Api

        Returns:
            A validated `LaunchProject` object.

        """
    if self.source == LaunchSource.DOCKER:
        return
    if self.source == LaunchSource.LOCAL:
        if not self._entry_point:
            wandb.termlog(f'{LOG_PREFIX}Entry point for repo not specified, defaulting to `python main.py`')
            self.set_entry_point(EntrypointDefaults.PYTHON)
    elif self.source == LaunchSource.JOB:
        self._fetch_job()
    else:
        self._fetch_project_local(internal_api=self.api)
    assert self.project_dir is not None
    if os.path.exists(os.path.join(self.project_dir, 'requirements.txt')) or os.path.exists(os.path.join(self.project_dir, 'requirements.frozen.txt')):
        self.deps_type = 'pip'
    elif os.path.exists(os.path.join(self.project_dir, 'environment.yml')):
        self.deps_type = 'conda'