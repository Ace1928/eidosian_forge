import logging
import os
import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT
def score_model_on_payload(model_uri, payload, eval_parameters=None):
    """Call the model identified by the given uri with the given payload."""
    if eval_parameters is None:
        eval_parameters = {}
    prefix, suffix = _parse_model_uri(model_uri)
    if prefix == 'openai':
        return _call_openai_api(suffix, payload, eval_parameters)
    elif prefix == 'gateway':
        return _call_gateway_api(suffix, payload, eval_parameters)
    elif prefix == 'endpoints':
        return _call_deployments_api(suffix, payload, eval_parameters)
    elif prefix in ('model', 'runs'):
        raise NotImplementedError
    else:
        raise MlflowException(f"Unknown model uri prefix '{prefix}'", error_code=INVALID_PARAMETER_VALUE)