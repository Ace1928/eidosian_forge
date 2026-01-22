from typing import Any, Dict, List, Optional
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.logging_utils import eprint
def pagination_wrapper_func(number_to_get, next_page_token):
    return MlflowClient().search_model_versions(max_results=number_to_get, filter_string=filter_string, order_by=order_by, page_token=next_page_token)