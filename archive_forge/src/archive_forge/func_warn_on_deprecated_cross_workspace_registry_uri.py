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
def warn_on_deprecated_cross_workspace_registry_uri(registry_uri):
    workspace_host, workspace_id = get_workspace_info_from_databricks_secrets(tracking_uri=registry_uri)
    if workspace_host is not None or workspace_id is not None:
        _logger.warning("Accessing remote workspace model registries using registry URIs of the form 'databricks://scope:prefix', or by loading models via URIs of the form 'models://scope:prefix@databricks/model-name/stage-or-version', is deprecated. Use Models in Unity Catalog instead for easy cross-workspace model access, with granular per-user audit logging and no extra setup required. See https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html for more details.")