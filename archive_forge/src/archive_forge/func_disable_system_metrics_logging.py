from mlflow.environment_variables import (
from mlflow.utils.annotations import experimental
@experimental
def disable_system_metrics_logging():
    """Disable system metrics logging globally.

    Calling this function will disable system metrics logging globally, but users can still opt in
    system metrics logging for individual runs by `mlflow.start_run(log_system_metrics=True)`.
    """
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(False)