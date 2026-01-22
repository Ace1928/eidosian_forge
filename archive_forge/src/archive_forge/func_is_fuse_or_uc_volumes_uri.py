import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def is_fuse_or_uc_volumes_uri(uri):
    """
    Validates whether a provided URI is directed to a FUSE mount point or a UC volumes mount point.
    Multiple directory paths are collapsed into a single designator for root path validation.
    For example, "////Volumes/" will resolve to "/Volumes/" for validation purposes.
    """
    resolved_uri = re.sub('/+', '/', uri)
    return any((resolved_uri.startswith(x) for x in [_DBFS_FUSE_PREFIX, _DBFS_HDFS_URI_PREFIX, _UC_VOLUMES_URI_PREFIX, _UC_DBFS_SYMLINK_PREFIX]))