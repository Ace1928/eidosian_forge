import os
from collections import OrderedDict
from itertools import zip_longest
from typing import List, Optional
from mlflow.entities import ExperimentTag, Metric, Param, RunStatus, RunTag, ViewType
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import GET_METRIC_HISTORY_MAX_RESULTS, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._tracking_service import utils
from mlflow.tracking.metric_value_conversion_utils import convert_metric_value_to_float_if_possible
from mlflow.utils import chunk_list
from mlflow.utils.async_logging.run_operations import RunOperations, get_combined_run_operations
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri
from mlflow.utils.validation import (
def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=True) -> Optional[RunOperations]:
    """Log multiple metrics, params, and/or tags.

        Args:
            run_id: String ID of the run.
            metrics: If provided, List of Metric(key, value, timestamp) instances.
            params: If provided, List of Param(key, value) instances.
            tags: If provided, List of RunTag(key, value) instances.
            synchronous: *Experimental* If True, blocks until the metrics/tags/params are logged
                successfully. If False, logs the metrics/tags/params asynchronously
                and returns a future representing the logging operation.

        Raises:
            MlflowException: If any errors occur.

        Returns:
            When synchronous=True, returns None.
            When synchronous=False, returns :py:class:`mlflow.RunOperations` that
            represents future for logging operation.

        """
    if len(metrics) == 0 and len(params) == 0 and (len(tags) == 0):
        return
    param_batches = chunk_list(params, MAX_PARAMS_TAGS_PER_BATCH)
    tag_batches = chunk_list(tags, MAX_PARAMS_TAGS_PER_BATCH)
    run_operations_list = []
    for params_batch, tags_batch in zip_longest(param_batches, tag_batches, fillvalue=[]):
        metrics_batch_size = min(MAX_ENTITIES_PER_BATCH - len(params_batch) - len(tags_batch), MAX_METRICS_PER_BATCH)
        metrics_batch_size = max(metrics_batch_size, 0)
        metrics_batch = metrics[:metrics_batch_size]
        metrics = metrics[metrics_batch_size:]
        if synchronous:
            self.store.log_batch(run_id=run_id, metrics=metrics_batch, params=params_batch, tags=tags_batch)
        else:
            run_operations_list.append(self.store.log_batch_async(run_id=run_id, metrics=metrics_batch, params=params_batch, tags=tags_batch))
    for metrics_batch in chunk_list(metrics, chunk_size=MAX_METRICS_PER_BATCH):
        if synchronous:
            self.store.log_batch(run_id=run_id, metrics=metrics_batch, params=[], tags=[])
        else:
            run_operations_list.append(self.store.log_batch_async(run_id=run_id, metrics=metrics_batch, params=[], tags=[]))
    if not synchronous:
        return get_combined_run_operations(run_operations_list)