import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
def create_package(directory: str, target_path: Path, include_parent_dir: bool=False, excludes: Optional[List[str]]=None, logger: Optional[logging.Logger]=default_logger):
    if excludes is None:
        excludes = []
    if logger is None:
        logger = default_logger
    if not target_path.exists():
        logger.info(f"Creating a file package for local directory '{directory}'.")
        _zip_directory(directory, excludes, target_path, include_parent_dir=include_parent_dir, logger=logger)