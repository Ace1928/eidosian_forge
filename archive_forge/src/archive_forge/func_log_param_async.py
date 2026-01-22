from abc import ABCMeta, abstractmethod
from typing import List, Optional
from mlflow.entities import DatasetInput, ViewType
from mlflow.entities.metric import MetricWithRunId
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations
def log_param_async(self, run_id, param) -> RunOperations:
    """
        Log a param for the specified run in async fashion.

        Args:
            run_id: String id for the run
            param: :py:class:`mlflow.entities.Param` instance to log
        """
    return self.log_batch_async(run_id, metrics=[], params=[param], tags=[])