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
def write_spark_dataframe_to_parquet_on_local_disk(spark_df, output_path):
    """Write spark dataframe in parquet format to local disk.

    Args:
        spark_df: Spark dataframe.
        output_path: Path to write the data to.

    """
    from mlflow.utils.databricks_utils import is_in_databricks_runtime
    if is_in_databricks_runtime():
        dbfs_path = os.path.join('.mlflow', 'cache', str(uuid.uuid4()))
        spark_df.coalesce(1).write.format('parquet').save(dbfs_path)
        shutil.copytree('/dbfs/' + dbfs_path, output_path)
        shutil.rmtree('/dbfs/' + dbfs_path)
    else:
        spark_df.coalesce(1).write.format('parquet').save(output_path)