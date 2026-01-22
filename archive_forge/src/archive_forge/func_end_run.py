import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def end_run(self, status: Optional[str]=None, run_id=None):
    """Terminates the run specified by run_id.

        If no ``run_id`` is passed in, then terminates the
        active run if one exists.

        Args:
            status (Optional[str]): The status to set when terminating the run.
            run_id (Optional[str]): The ID of the run to terminate.

        """
    if run_id and self._run_exists(run_id) and (not (self._mlflow.active_run() and self._mlflow.active_run().info.run_id == run_id)):
        client = self._get_client()
        client.set_terminated(run_id=run_id, status=status)
    else:
        self._mlflow.end_run(status=status)