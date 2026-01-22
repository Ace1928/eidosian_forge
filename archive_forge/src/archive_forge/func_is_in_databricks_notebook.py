import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
@_use_repl_context_if_available('isInNotebook')
def is_in_databricks_notebook():
    if _get_property_from_spark_context('spark.databricks.notebook.id') is not None:
        return True
    try:
        return acl_path_of_acl_root().startswith('/workspace')
    except Exception:
        return False