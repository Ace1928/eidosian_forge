import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_client_gateway(self, client_gateway_id: str=None, dry_run: bool=False):
    """
        Deletes a client gateway.
        You must delete the VPN connection before deleting the client gateway.

        :param      client_gateway_id: The ID of the client gateway
        you want to delete. (required)
        :type       client_gateway_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns True if action is successful
        :rtype: ``bool``
        """
    action = 'DeleteClientGateway'
    data = {'DryRun': dry_run}
    if client_gateway_id is not None:
        data.update({'ClientGatewayId': client_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()