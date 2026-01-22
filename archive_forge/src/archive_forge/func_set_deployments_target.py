import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def set_deployments_target(target: str):
    """Sets the target deployment client for MLflow deployments

    Args:
        target: The full uri of a running MLflow deployments server or, if running on
            Databricks, "databricks".
    """
    if not _is_valid_target(target):
        raise MlflowException.invalid_parameter_value("The target provided is not a valid uri or 'databricks'")
    global _deployments_target
    _deployments_target = target