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
def get_model_dependency_oauth_token(should_retry=True):
    try:
        with open(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH) as f:
            oauth_dict = json.load(f)
            return oauth_dict['OAUTH_TOKEN'][0]['oauthTokenValue']
    except Exception as e:
        if should_retry:
            time.sleep(0.5)
            return get_model_dependency_oauth_token(should_retry=False)
        else:
            raise MlflowException('Unable to read Oauth credentials from file mount for Databricks Model Serving dependency failed') from e