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
def json_encode(self) -> Dict[str, Any]:
    """Returns the artifact encoded to the JSON format.

        Returns:
            A `dict` with `string` keys representing attributes of the artifact.
        """
    self._ensure_logged('json_encode')
    return util.artifact_to_json(self)