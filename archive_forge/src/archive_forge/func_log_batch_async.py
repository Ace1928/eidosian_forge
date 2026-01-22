from abc import ABCMeta, abstractmethod
from typing import List, Optional
from mlflow.entities import DatasetInput, ViewType
from mlflow.entities.metric import MetricWithRunId
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations
def log_batch_async(self, run_id, metrics, params, tags) -> RunOperations:
    """
        Log multiple metrics, params, and tags for the specified run in async fashion.
        This API does not offer immediate consistency of the data. When API returns,
        data is accepted but not persisted/processed by back end. Data would be processed
        in near real time fashion.

        Args:
            run_id: String id for the run.
            metrics: List of :py:class:`mlflow.entities.Metric` instances to log.
            params: List of :py:class:`mlflow.entities.Param` instances to log.
            tags: List of :py:class:`mlflow.entities.RunTag` instances to log.

        Returns:
            An :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance
            that represents future for logging operation.
        """
    if not self._async_logging_queue.is_active():
        self._async_logging_queue.activate()
    return self._async_logging_queue.log_batch_async(run_id=run_id, metrics=metrics, params=params, tags=tags)