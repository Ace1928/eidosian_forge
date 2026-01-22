import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
def resolve_dataset_source(raw_source: Any, candidate_sources: Optional[List[DatasetSource]]=None) -> DatasetSource:
    """Resolves a raw source object, such as a string URI, to a DatasetSource for use with
    MLflow Tracking.

    Args:
        raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data" or a
            HuggingFace :py:class:`datasets.Dataset` object.
        candidate_sources: A list of DatasetSource classes to consider as potential sources
            when resolving the raw source. Subclasses of the specified candidate
            sources are also considered. If unspecified, all registered sources
            are considered.

    Raises:
        MlflowException: If no DatasetSource class can resolve the raw source.

    Returns:
        The resolved DatasetSource.
    """
    return _dataset_source_registry.resolve(raw_source=raw_source, candidate_sources=candidate_sources)