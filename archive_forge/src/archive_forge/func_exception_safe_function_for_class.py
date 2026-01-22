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
def exception_safe_function_for_class(function):
    """
    Wraps the specified function with broad exception handling to guard
    against unexpected errors during autologging.
    Note this function creates an unpicklable function as `safe_function` is locally defined,
    but a class instance containing methods decorated by this function should be pickalable,
    because pickle only saves instance attributes, not methods.
    See https://docs.python.org/3/library/pickle.html#pickling-class-instances for more details.
    """
    if is_testing():
        setattr(function, _ATTRIBUTE_EXCEPTION_SAFE, True)

    def safe_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            if is_testing():
                raise
            else:
                _logger.warning('Encountered unexpected error during autologging: %s', e)
    return update_wrapper_extended(safe_function, function)