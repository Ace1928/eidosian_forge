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
def is_valid_dbfs_uri(uri):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != 'dbfs':
        return False
    try:
        db_profile_uri = get_databricks_profile_uri_from_artifact_uri(uri)
    except MlflowException:
        db_profile_uri = None
    return not parsed.netloc or db_profile_uri is not None