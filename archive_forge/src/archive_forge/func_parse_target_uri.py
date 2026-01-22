import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def parse_target_uri(target_uri):
    """Parse out the deployment target from the provided target uri"""
    parsed = urllib.parse.urlparse(target_uri)
    if not parsed.scheme:
        if parsed.path:
            return parsed.path
        raise MlflowException(f'Not a proper deployment URI: {target_uri}. ' + "Deployment URIs must be of the form 'target' or 'target:/suffix'")
    return parsed.scheme