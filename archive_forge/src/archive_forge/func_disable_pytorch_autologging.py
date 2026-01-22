import time
from contextlib import contextmanager
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
@contextmanager
def disable_pytorch_autologging():
    global DISABLED
    old_value = DISABLED
    DISABLED = True
    try:
        yield
    finally:
        DISABLED = old_value