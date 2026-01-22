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
def download_chunk_retries(*, chunks, http_uri, headers, download_path):
    num_retries = _MLFLOW_MPD_NUM_RETRIES.get()
    interval = _MLFLOW_MPD_RETRY_INTERVAL_SECONDS.get()
    for chunk in chunks:
        _logger.info(f'Retrying download of chunk {chunk.index} for {chunk.path}')
        for retry in range(num_retries):
            try:
                download_chunk(range_start=chunk.start, range_end=chunk.end, headers=headers, download_path=download_path, http_uri=http_uri)
                _logger.info(f'Successfully downloaded chunk {chunk.index} for {chunk.path}')
                break
            except Exception:
                if retry == num_retries - 1:
                    raise
            time.sleep(interval)