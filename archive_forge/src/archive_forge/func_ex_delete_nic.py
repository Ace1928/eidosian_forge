import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_nic(self, nic_id: str=None, dry_run: bool=False):
    """
        Deletes the specified network interface card (NIC).
        The network interface must not be attached to any virtual machine (VM).

        :param      nic_id: The ID of the NIC you want to delete. (required)
        :type       nic_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteNic'
    data = {'DryRun': dry_run}
    if nic_id is not None:
        data.update({'NicId': nic_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()