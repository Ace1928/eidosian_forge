import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_snapshot_export_tasks(self, dry_run: bool=False, task_ids: List[str]=None):
    """
        Lists one or more image export tasks.

        :param      task_ids: The IDs of the export tasks.
        :type       task_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: snapshot export tasks
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadSnapshotExportTasks'
    data = {'DryRun': dry_run, 'Filters': {}}
    if task_ids is not None:
        data['Filters'].update({'TaskIds': task_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['SnapshotExportTasks']
    return response.json()