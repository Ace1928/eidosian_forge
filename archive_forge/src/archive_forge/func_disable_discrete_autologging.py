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
@contextlib.contextmanager
def disable_discrete_autologging(flavors_to_disable: List[str]) -> None:
    """
    Context manager for disabling specific autologging integrations temporarily while another
    flavor's autologging is activated. This context wrapper is useful in the event that, for
    example, a particular library calls upon another library within a training API that has a
    current MLflow autologging integration.
    For instance, the transformers library's Trainer class, when running metric scoring,
    builds a sklearn model and runs evaluations as part of its accuracy scoring. Without this
    temporary autologging disabling, a new run will be generated that contains a sklearn model
    that holds no use for tracking purposes as it is only used during the metric evaluation phase
    of training.

    Args:
        flavors_to_disable: A list of flavors that need to be temporarily disabled while
            executing another flavor's autologging to prevent spurious run
            logging of unrelated models, metrics, and parameters.
    """
    enabled_flavors = []
    for flavor in flavors_to_disable:
        if not autologging_is_disabled(flavor):
            enabled_flavors.append(flavor)
            autolog_func = getattr(mlflow, flavor)
            autolog_func.autolog(disable=True)
    yield
    for flavor in enabled_flavors:
        autolog_func = getattr(mlflow, flavor)
        autolog_func.autolog(disable=False)