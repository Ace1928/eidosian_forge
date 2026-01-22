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
def get_workspace_info_from_dbutils():
    try:
        dbutils = _get_dbutils()
        if dbutils:
            browser_hostname = get_browser_hostname()
            workspace_host = 'https://' + browser_hostname if browser_hostname else get_webapp_url()
            workspace_id = get_workspace_id()
            return (workspace_host, workspace_id)
    except Exception:
        pass
    return (None, None)