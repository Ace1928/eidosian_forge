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
def package_exists(pkg_uri: str) -> bool:
    """Check whether the package with given URI exists or not.

    Args:
        pkg_uri: The uri of the package

    Return:
        True for package existing and False for not.
    """
    protocol, pkg_name = parse_uri(pkg_uri)
    if protocol == Protocol.GCS:
        return _internal_kv_exists(pkg_uri)
    else:
        raise NotImplementedError(f'Protocol {protocol} is not supported')