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
def to_environment(self):
    env = {DatabricksWorkspaceInfo.WORKSPACE_HOST_ENV_VAR: self.host}
    if self.workspace_id is not None:
        env[DatabricksWorkspaceInfo.WORKSPACE_ID_ENV_VAR] = self.workspace_id
    return env