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
def validate_db_scope_prefix_info(scope, prefix):
    for c in ['/', ':', ' ']:
        if c in scope:
            raise MlflowException(f"Unsupported Databricks profile name: {scope}. Profile names cannot contain '{c}'.")
        if prefix and c in prefix:
            raise MlflowException(f"Unsupported Databricks profile key prefix: {prefix}. Key prefixes cannot contain '{c}'.")
    if prefix is not None and prefix.strip() == '':
        raise MlflowException(f"Unsupported Databricks profile key prefix: '{prefix}'. Key prefixes cannot be empty.")