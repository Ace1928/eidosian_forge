import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.async_logging.run_batch import RunBatch
from mlflow.utils.async_logging.run_operations import RunOperations
def logging_func(run_batch):
    try:
        self._logging_func(run_id=run_batch.run_id, metrics=run_batch.metrics, params=run_batch.params, tags=run_batch.tags)
        run_batch.completion_event.set()
    except Exception as e:
        _logger.error(f'Run Id {run_batch.run_id}: Failed to log run data: Exception: {e}')
        run_batch.exception = e
        run_batch.completion_event.set()