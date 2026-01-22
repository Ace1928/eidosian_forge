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
def remove_databricks_profile_info_from_artifact_uri(artifact_uri):
    """
    Only removes the netloc portion of the URI if it is a Databricks
    profile specification, e.g.
    ``profile@databricks`` or ``secret_scope:key_prefix@databricks``.
    """
    parsed = urllib.parse.urlparse(artifact_uri)
    if not parsed.netloc or parsed.hostname != 'databricks':
        return artifact_uri
    return urllib.parse.urlunparse(parsed._replace(netloc=''))