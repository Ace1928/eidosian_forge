from abc import ABCMeta, abstractmethod
from typing import List, Optional
from mlflow.entities import DatasetInput, ViewType
from mlflow.entities.metric import MetricWithRunId
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations
def flush_async_logging(self):
    """
        Flushes the async logging queue. This method is a no-op if the queue is not active.
        """
    if self._async_logging_queue.is_active():
        self._async_logging_queue.flush()