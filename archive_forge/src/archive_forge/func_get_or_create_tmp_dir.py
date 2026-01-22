import atexit
import codecs
import errno
import fnmatch
import gzip
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import Optional, Union
from urllib.parse import unquote
from urllib.request import pathname2url
import yaml
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils import download_cloud_file_chunk, merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status
@cache_return_value_per_process
def get_or_create_tmp_dir():
    """
    Get or create a temporary directory which will be removed once python process exit.
    """
    from mlflow.utils.databricks_utils import get_repl_id, is_in_databricks_runtime
    if is_in_databricks_runtime() and get_repl_id() is not None:
        try:
            repl_local_tmp_dir = _get_dbutils().entry_point.getReplLocalTempDir()
        except Exception:
            repl_local_tmp_dir = os.path.join('/tmp', 'repl_tmp_data', get_repl_id())
        tmp_dir = os.path.join(repl_local_tmp_dir, 'mlflow')
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp()
        os.chmod(tmp_dir, 511)
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    return tmp_dir