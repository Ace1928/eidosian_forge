import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_net(self, net_id: str=None, dhcp_options_set_id: str=None, dry_run: bool=False):
    """
        Associates a DHCP options set with a specified Net.

        :param      net_id: The ID of the Net. (required)
        :type       net_id: ``str``

        :param      dhcp_options_set_id: The ID of the DHCP options set
        (or default if you want to associate the default one). (required)
        :type       dhcp_options_set_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The modified Nat Service
        :rtype: ``dict``
        """
    action = 'UpdateNet'
    data = {'DryRun': dry_run}
    if net_id is not None:
        data.update({'NetId': net_id})
    if dhcp_options_set_id is not None:
        data.update({'DhcpOptionsSetId': dhcp_options_set_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Net']
    return response.json()