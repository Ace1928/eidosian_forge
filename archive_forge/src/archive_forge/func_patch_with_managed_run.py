import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def patch_with_managed_run(original, *args, **kwargs):
    managed_run = None
    if not mlflow.active_run():
        managed_run = create_managed_run()
    try:
        result = patch_function(original, *args, **kwargs)
    except (Exception, KeyboardInterrupt):
        if managed_run:
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
        raise
    else:
        if managed_run:
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
        return result