import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def start_run(self, run_name: Optional[str]=None, tags: Optional[Dict]=None, set_active: bool=False) -> 'Run':
    """Starts a new run and possibly sets it as the active run.

        Args:
            tags: Tags to set for the new run.
            set_active: Whether to set the new run as the active run.
                If an active run already exists, then that run is returned.

        Returns:
            The newly created MLflow run.
        """
    import mlflow
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
    if tags is None:
        tags = {}
    if set_active:
        return self._start_active_run(run_name=run_name, tags=tags)
    client = self._get_client()
    if version.parse(mlflow.__version__) >= version.parse('1.30.0'):
        run = client.create_run(run_name=run_name, experiment_id=self.experiment_id, tags=tags)
    else:
        tags[MLFLOW_RUN_NAME] = run_name
        run = client.create_run(experiment_id=self.experiment_id, tags=tags)
    return run