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
def upload_package_to_gcs(pkg_uri: str, pkg_bytes: bytes) -> None:
    """Upload a local package to GCS.

    Args:
        pkg_uri: The URI of the package, e.g. gcs://my_package.zip
        pkg_bytes: The data to be uploaded.

    Raises:
        RuntimeError: If the upload fails.
        ValueError: If the pkg_uri is a remote path or if the data's
            size exceeds GCS_STORAGE_MAX_SIZE.
        NotImplementedError: If the protocol of the URI is not supported.

    """
    protocol, pkg_name = parse_uri(pkg_uri)
    if protocol == Protocol.GCS:
        _store_package_in_gcs(pkg_uri, pkg_bytes)
    elif protocol in Protocol.remote_protocols():
        raise ValueError('upload_package_to_gcs should not be called with a remote path.')
    else:
        raise NotImplementedError(f'Protocol {protocol} is not supported')