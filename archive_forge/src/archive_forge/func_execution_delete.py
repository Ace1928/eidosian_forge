import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def execution_delete(self, execution, mistral_client=None):
    """Remove a given schedule execution.

        :param id: id of execution to remove.
        """
    exec_id = execution.id if isinstance(execution, ScheduleExecution) else execution
    if isinstance(execution, ScheduleExecution):
        execution = execution.name
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    mistral_client.executions.delete(exec_id)