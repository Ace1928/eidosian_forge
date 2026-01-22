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
def get_databricks_workspace_info_from_uri(tracking_uri: str) -> Optional[DatabricksWorkspaceInfo]:
    if not is_databricks_uri(tracking_uri):
        return None
    if is_databricks_default_tracking_uri(tracking_uri) and (is_in_databricks_notebook() or is_in_databricks_job()):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
    else:
        workspace_host, workspace_id = get_workspace_info_from_databricks_secrets(tracking_uri)
        if not workspace_id:
            _logger.info('No workspace ID specified; if your Databricks workspaces share the same host URL, you may want to specify the workspace ID (along with the host information in the secret manager) for run lineage tracking. For more details on how to specify this information in the secret manager, please refer to the Databricks MLflow documentation.')
    if workspace_host:
        return DatabricksWorkspaceInfo(host=workspace_host, workspace_id=workspace_id)
    else:
        return None