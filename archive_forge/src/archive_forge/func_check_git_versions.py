import atexit
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from .hf_api import HfApi, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import (
from .utils._deprecation import _deprecate_method
def check_git_versions(self):
    """
        Checks that `git` and `git-lfs` can be run.

        Raises:
            - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
              if `git` or `git-lfs` are not installed.
        """
    try:
        git_version = run_subprocess('git --version', self.local_dir).stdout.strip()
    except FileNotFoundError:
        raise EnvironmentError('Looks like you do not have git installed, please install.')
    try:
        lfs_version = run_subprocess('git-lfs --version', self.local_dir).stdout.strip()
    except FileNotFoundError:
        raise EnvironmentError('Looks like you do not have git-lfs installed, please install. You can install from https://git-lfs.github.com/. Then run `git lfs install` (you only have to do this once).')
    logger.info(git_version + '\n' + lfs_version)