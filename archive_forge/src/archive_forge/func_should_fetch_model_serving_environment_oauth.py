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
def should_fetch_model_serving_environment_oauth():
    return is_in_databricks_model_serving_environment() and os.path.exists(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH) and os.path.isfile(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH)