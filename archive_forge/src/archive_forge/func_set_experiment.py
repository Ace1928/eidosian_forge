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
def set_experiment(experiment_name: Optional[str]=None, experiment_id: Optional[str]=None) -> Experiment:
    """
    Set the given experiment as the active experiment. The experiment must either be specified by
    name via `experiment_name` or by ID via `experiment_id`. The experiment name and ID cannot
    both be specified.

    .. note::
        If the experiment being set by name does not exist, a new experiment will be
        created with the given name. After the experiment has been created, it will be set
        as the active experiment. On certain platforms, such as Databricks, the experiment name
        must be an absolute path, e.g. ``"/Users/<username>/my-experiment"``.

    Args:
        experiment_name: Case sensitive name of the experiment to be activated.
        experiment_id: ID of the experiment to be activated. If an experiment with this ID
            does not exist, an exception is thrown.

    Returns:
        An instance of :py:class:`mlflow.entities.Experiment` representing the new active
        experiment.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow

        # Set an experiment name, which must be unique and case-sensitive.
        experiment = mlflow.set_experiment("Social NLP Experiments")
        # Get Experiment Details
        print(f"Experiment_id: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Tags: {experiment.tags}")
        print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    .. code-block:: text
        :caption: Output

        Experiment_id: 1
        Artifact Location: file:///.../mlruns/1
        Tags: {}
        Lifecycle_stage: active
    """
    if experiment_name is not None and experiment_id is not None or (experiment_name is None and experiment_id is None):
        raise MlflowException(message='Must specify exactly one of: `experiment_id` or `experiment_name`.', error_code=INVALID_PARAMETER_VALUE)
    client = MlflowClient()
    if experiment_id is None:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            _logger.info("Experiment with name '%s' does not exist. Creating a new experiment.", experiment_name)
            experiment_id = client.create_experiment(experiment_name)
            experiment = client.get_experiment(experiment_id)
    else:
        experiment = client.get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(message=f"Experiment with ID '{experiment_id}' does not exist.", error_code=RESOURCE_DOES_NOT_EXIST)
    if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
        raise MlflowException(message="Cannot set a deleted experiment '%s' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one." % experiment.name, error_code=INVALID_PARAMETER_VALUE)
    global _active_experiment_id
    _active_experiment_id = experiment.experiment_id
    return experiment