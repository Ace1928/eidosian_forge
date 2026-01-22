import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_flexible_gpu(self, flexible_gpu_id: str=None, dry_run: bool=False):
    """
        Releases a flexible GPU (fGPU) from your account.
        The fGPU becomes free to be used by someone else.

        :param      flexible_gpu_id: The ID of the fGPU you want
        to delete. (required)
        :type       flexible_gpu_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteFlexibleGpu'
    data = {'DryRun': dry_run}
    if flexible_gpu_id is not None:
        data.update({'FlexibleGpuId': flexible_gpu_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()