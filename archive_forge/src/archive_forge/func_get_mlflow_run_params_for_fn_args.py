import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def get_mlflow_run_params_for_fn_args(fn, args, kwargs, unlogged=None):
    """Given arguments explicitly passed to a function, generate a dictionary of MLflow Run
    parameter key / value pairs.

    Args:
        fn: function whose parameters are to be logged.
        args: arguments explicitly passed into fn. If `fn` is defined on a class,
            `self` should not be part of `args`; the caller is responsible for
            filtering out `self` before calling this function.
        kwargs: kwargs explicitly passed into fn.
        unlogged: parameters not to be logged.

    Returns:
        A dictionary of MLflow Run parameter key / value pairs.
    """
    unlogged = unlogged or []
    param_spec = inspect.signature(fn).parameters
    relevant_params = [param for param in param_spec.values() if param.name != 'self']
    params_to_log = {param_info.name: param_val for param_info, param_val in zip(list(relevant_params)[:len(args)], args)}
    params_to_log.update(kwargs)
    params_to_log.update({param.name: param.default for param in list(relevant_params)[len(args):] if param.name not in kwargs})
    return {key: value for key, value in params_to_log.items() if key not in unlogged}