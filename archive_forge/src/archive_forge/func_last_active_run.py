import atexit
import contextlib
import importlib
import inspect
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import _get_store, artifact_utils
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.default_experiment import registry as default_experiment_registry
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.mlflow_tags import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import _validate_experiment_id_type, _validate_run_id
def last_active_run() -> Optional[Run]:
    """Gets the most recent active run.

    Examples:

    .. code-block:: python
        :test:
        :caption: To retrieve the most recent autologged run:

        import mlflow

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor

        mlflow.autolog()

        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        # Create and train models.
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)

        # Use the model to make predictions on the test dataset.
        predictions = rf.predict(X_test)
        autolog_run = mlflow.last_active_run()

    .. code-block:: python
        :test:
        :caption: To get the most recently active run that ended:

        import mlflow

        mlflow.start_run()
        mlflow.end_run()
        run = mlflow.last_active_run()

    .. code-block:: python
        :test:
        :caption: To retrieve the currently active run:

        import mlflow

        mlflow.start_run()
        run = mlflow.last_active_run()
        mlflow.end_run()

    Returns:
        The active run (this is equivalent to ``mlflow.active_run()``) if one exists.
        Otherwise, the last run started from the current Python process that reached
        a terminal status (i.e. FINISHED, FAILED, or KILLED).
    """
    _active_run = active_run()
    if _active_run is not None:
        return _active_run
    if _last_active_run_id is None:
        return None
    return get_run(_last_active_run_id)