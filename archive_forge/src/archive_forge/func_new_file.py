import atexit
import concurrent.futures
import contextlib
import json
import multiprocessing.dummy
import os
import re
import shutil
import tempfile
import time
from copy import copy
from datetime import datetime, timedelta
from functools import partial
from pathlib import PurePosixPath
from typing import (
from urllib.parse import urlparse
import requests
import wandb
from wandb import data_types, env, util
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public import ArtifactCollection, ArtifactFiles, RetryingClient, Run
from wandb.data_types import WBValue
from wandb.errors.term import termerror, termlog, termwarn
from wandb.sdk.artifacts.artifact_download_logger import ArtifactDownloadLogger
from wandb.sdk.artifacts.artifact_instance_cache import artifact_instance_cache
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.artifact_manifests.artifact_manifest_v1 import (
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.artifacts.artifact_ttl import ArtifactTTL
from wandb.sdk.artifacts.exceptions import (
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.artifacts.storage_layout import StorageLayout
from wandb.sdk.artifacts.storage_policies import WANDB_STORAGE_POLICY
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.data_types._dtypes import Type as WBType
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, retry, runid, telemetry
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, md5_file_b64
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
from wandb.sdk.lib.runid import generate_id
from wandb.util import get_core_path
from wandb_gql import gql  # noqa: E402
@contextlib.contextmanager
def new_file(self, name: str, mode: str='w', encoding: Optional[str]=None) -> Generator[IO, None, None]:
    """Open a new temporary file and add it to the artifact.

        Arguments:
            name: The name of the new file to add to the artifact.
            mode: The file access mode to use to open the new file.
            encoding: The encoding used to open the new file.

        Returns:
            A new file object that can be written to. Upon closing, the file will be
            automatically added to the artifact.

        Raises:
            ArtifactFinalizedError: You cannot make changes to the current artifact
            version because it is finalized. Log a new artifact version instead.
        """
    self._ensure_can_add()
    if self._tmp_dir is None:
        self._tmp_dir = tempfile.TemporaryDirectory()
    path = os.path.join(self._tmp_dir.name, name.lstrip('/'))
    if os.path.exists(path):
        raise ValueError(f'File with name {name!r} already exists at {path!r}')
    filesystem.mkdir_exists_ok(os.path.dirname(path))
    try:
        with util.fsync_open(path, mode, encoding) as f:
            yield f
    except UnicodeEncodeError as e:
        termerror(f'Failed to open the provided file (UnicodeEncodeError: {e}). Please provide the proper encoding.')
        raise e
    self.add_file(path, name=name)