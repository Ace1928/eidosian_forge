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
def resolve_uri_if_local(local_uri):
    """
    if `local_uri` is passed in as a relative local path, this function
    resolves it to absolute path relative to current working directory.

    Args:
        local_uri: Relative or absolute path or local file uri

    Returns:
        a fully-formed absolute uri path or an absolute filesystem path
    """
    from mlflow.utils.file_utils import local_file_uri_to_path
    if local_uri is not None and is_local_uri(local_uri):
        scheme = get_uri_scheme(local_uri)
        cwd = pathlib.Path.cwd()
        local_path = local_file_uri_to_path(local_uri)
        if not pathlib.Path(local_path).is_absolute():
            if scheme == '':
                if is_windows():
                    return urllib.parse.urlunsplit(('file', None, cwd.joinpath(local_path).as_posix(), None, None))
                return cwd.joinpath(local_path).as_posix()
            local_uri_split = urllib.parse.urlsplit(local_uri)
            return urllib.parse.urlunsplit((local_uri_split.scheme, None, cwd.joinpath(local_path).as_posix(), local_uri_split.query, local_uri_split.fragment))
    return local_uri