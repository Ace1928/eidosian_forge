import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import (
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema
def from_tensorflow(features, source: Optional[Union[str, DatasetSource]]=None, targets=None, name: Optional[str]=None, digest: Optional[str]=None) -> TensorFlowDataset:
    """Constructs a TensorFlowDataset object from TensorFlow data, optional targets, and source.

    If the source is path like, then this will construct a DatasetSource object from the source
    path. Otherwise, the source is assumed to be a DatasetSource object.

    Args:
        features: A TensorFlow dataset or tensor of features.
        source: The source from which the data was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. If source is not a path like string,
            pass in a DatasetSource object directly. If no source is specified,
            a CodeDatasetSource is used, which will source information from the run
            context.
        targets: A TensorFlow dataset or tensor of targets. Optional.
        name: The name of the dataset. If unspecified, a name is generated.
        digest: A dataset digest (hash). If unspecified, a digest is computed
            automatically.
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry
    if source is not None:
        if isinstance(source, DatasetSource):
            resolved_source = source
        else:
            resolved_source = resolve_dataset_source(source)
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)
    return TensorFlowDataset(features=features, source=resolved_source, targets=targets, name=name, digest=digest)