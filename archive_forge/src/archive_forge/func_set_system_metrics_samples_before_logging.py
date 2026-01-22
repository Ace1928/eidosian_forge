from mlflow.environment_variables import (
from mlflow.utils.annotations import experimental
@experimental
def set_system_metrics_samples_before_logging(samples):
    """Set the number of samples before logging system metrics.

    Every time `samples` samples have been collected, the system metrics will be logged to mlflow.
    By default `samples=1`.
    """
    if samples is None:
        MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.unset()
    else:
        MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.set(samples)