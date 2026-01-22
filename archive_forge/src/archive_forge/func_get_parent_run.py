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
def get_parent_run(run_id: str) -> Optional[Run]:
    """Gets the parent run for the given run id if one exists.

    Args:
        run_id: Unique identifier for the child run.

    Returns:
        A single :py:class:`mlflow.entities.Run` object, if the parent run exists. Otherwise,
        returns None.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow

        # Create nested runs
        with mlflow.start_run():
            with mlflow.start_run(nested=True) as child_run:
                child_run_id = child_run.info.run_id

        parent_run = mlflow.get_parent_run(child_run_id)

        print(f"child_run_id: {child_run_id}")
        print(f"parent_run_id: {parent_run.info.run_id}")

    .. code-block:: text
        :caption: Output

        child_run_id: 7d175204675e40328e46d9a6a5a7ee6a
        parent_run_id: 8979459433a24a52ab3be87a229a9cdf
    """
    return MlflowClient().get_parent_run(run_id)