import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_vpn_connection_route(self, vpn_connection_id: str=None, destination_ip_range: str=None, dry_run: bool=False):
    """
        Deletes a specified VPN connection.
        If you want to delete a Net and all its dependencies, we recommend to
        detach the virtual gateway from the Net and delete the Net before
        deleting the VPN connection. This enables you to delete the Net
        without waiting for the VPN connection to be deleted.

        :param      vpn_connection_id: the ID of the VPN connection you want
        to delete.
        (required)
        :type       vpn_connection_id: ``str``

        :param      destination_ip_range: The network prefix of the route to
        delete, in CIDR notation (for example, 10.12.0.0/16). (required)
        :type       destination_ip_range: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteVpnConnectionRoute'
    data = {'DryRun': dry_run}
    if vpn_connection_id is not None:
        data.update({'VpnConnectionId': vpn_connection_id})
    if destination_ip_range is not None:
        data.update({'DestinationIpRange': destination_ip_range})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()