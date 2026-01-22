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
def check_databricks_secret_scope_access(scope_name):
    dbutils = _get_dbutils()
    if dbutils:
        try:
            dbutils.secrets.list(scope_name)
        except Exception as e:
            _logger.warning(f"Unable to access Databricks secret scope '{scope_name}' for OpenAI credentials that will be used to deploy the model to Databricks Model Serving. Please verify that the current Databricks user has 'READ' permission for this scope. For more information, see https://mlflow.org/docs/latest/python_api/openai/index.html#credential-management-for-openai-on-databricks. Error: {e}")