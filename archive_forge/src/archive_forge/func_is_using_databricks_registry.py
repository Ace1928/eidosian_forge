import urllib.parse
from typing import NamedTuple, Optional
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri
def is_using_databricks_registry(uri):
    profile_uri = get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
    return is_databricks_uri(profile_uri)