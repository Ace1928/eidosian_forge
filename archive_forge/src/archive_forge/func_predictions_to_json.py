import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from typing import Dict, NamedTuple, Optional, Tuple
import flask
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature
from mlflow.types import ParamSchema, Schema
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.proto_json_utils import (
from mlflow.version import VERSION
from io import StringIO
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.server.handlers import catch_mlflow_exception
def predictions_to_json(raw_predictions, output, metadata=None):
    if metadata and 'predictions' in metadata:
        raise MlflowException("metadata cannot contain 'predictions' key", error_code=INVALID_PARAMETER_VALUE)
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient='records')
    return json.dump({'predictions': predictions, **(metadata or {})}, output, cls=NumpyEncoder)