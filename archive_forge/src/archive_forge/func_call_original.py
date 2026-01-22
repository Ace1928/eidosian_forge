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
def call_original(*og_args, **og_kwargs):

    def _original_fn(*_og_args, **_og_kwargs):
        if is_testing():
            _validate_args(autologging_integration, function_name, args, kwargs, og_args, og_kwargs)
            nonlocal patch_function_run_for_testing
            patch_function_run_for_testing = mlflow.active_run()
        nonlocal original_has_been_called
        original_has_been_called = True
        nonlocal original_result
        with set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings=False, reroute_warnings=False):
            original_result = original(*_og_args, **_og_kwargs)
            return original_result
    return call_original_fn_with_event_logging(_original_fn, og_args, og_kwargs)