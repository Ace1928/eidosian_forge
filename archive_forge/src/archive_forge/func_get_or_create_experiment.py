import os
from typing import Any, Dict, Optional, Union
import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
def get_or_create_experiment(name: Optional[str]=None, tracking_uri: Optional[str]=None, registry_uri: Optional[str]=None) -> 'MLFlowExperimentLevelLogger':
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    if name is None:
        eid = _get_experiment_id()
        exp = client.get_experiment(eid)
    else:
        try:
            eid = client.create_experiment(name)
            exp = client.get_experiment(eid)
        except MlflowException as e:
            if e.error_code != 'RESOURCE_ALREADY_EXISTS':
                raise
            exp = client.get_experiment_by_name(name)
    return MLFlowExperimentLevelLogger(client, exp)