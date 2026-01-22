import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_net_peering(self, net_peering_id: List[str]=None, dry_run: bool=False):
    """
        Deletes a Net peering connection.
        If the Net peering connection is in the active state, it can be
        deleted either by the owner of the requester Net or the owner of the
        peer Net.
        If it is in the pending-acceptance state, it can be deleted only by
        the owner of the requester Net.
        If it is in the rejected, failed, or expired states, it cannot be
        deleted.

        :param      net_peering_id: The ID of the Net peering connection you
        want to delete. (required)
        :type       net_peering_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteNetPeering'
    data = {'DryRun': dry_run}
    if net_peering_id is not None:
        data.update({'NetPeeringId': net_peering_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()