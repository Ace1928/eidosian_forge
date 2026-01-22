import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
def get_registered_sources() -> List[DatasetSource]:
    """Obtains the registered dataset sources.

    Returns:
        A list of registered dataset sources.

    """
    return _dataset_source_registry.sources