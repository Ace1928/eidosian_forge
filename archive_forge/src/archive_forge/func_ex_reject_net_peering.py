import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_reject_net_peering(self, net_peering_id: List[str]=None, dry_run: bool=False):
    """
        Rejects a Net peering connection request.
        The Net peering connection must be in the pending-acceptance state to
        be rejected. The rejected Net peering connection is then in the
        rejected state.

        :param      net_peering_id: The ID of the Net peering connection you
        want to reject. (required)
        :type       net_peering_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The rejected Net Peering
        :rtype: ``dict``
        """
    action = 'RejectNetPeering'
    data = {'DryRun': dry_run}
    if net_peering_id is not None:
        data.update({'NetPeeringId': net_peering_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()