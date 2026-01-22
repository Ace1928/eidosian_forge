import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_unlink_route_table(self, link_route_table_id: str=None, dry_run: bool=False):
    """
        Disassociates a Subnet from a route table.
        After disassociation, the Subnet can no longer use the routes in this
        route table, but uses the routes in the main route table of the Net
        instead.

        :param      link_route_table_id: The ID of the route table. (required)
        :type       link_route_table_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'UnlinkRouteTable'
    data = {'DryRun': dry_run}
    if link_route_table_id is not None:
        data.update({'LinkRouteTableId': link_route_table_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()