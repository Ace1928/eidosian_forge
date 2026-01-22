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
def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
    try:
        try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_start, session, destination, function_name, og_args, og_kwargs)
        original_fn_result = original_fn(*og_args, **og_kwargs)
        try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_success, session, destination, function_name, og_args, og_kwargs)
        return original_fn_result
    except Exception as original_fn_e:
        try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_error, session, destination, function_name, og_args, og_kwargs, original_fn_e)
        nonlocal failed_during_original
        failed_during_original = True
        raise