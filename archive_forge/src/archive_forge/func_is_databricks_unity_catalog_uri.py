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
def is_databricks_unity_catalog_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == _DATABRICKS_UNITY_CATALOG_SCHEME or uri == _DATABRICKS_UNITY_CATALOG_SCHEME