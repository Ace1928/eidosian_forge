import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_net_access_point(self, add_route_table_ids: List[str]=None, net_access_point_id: str=None, remove_route_table_ids: List[str]=None, dry_run: bool=False):
    """
        Modifies the attributes of a Net access point.
        This action enables you to add or remove route tables associated with
        the specified Net access point.

        :param      add_route_table_ids: One or more IDs of route tables to
        associate with the specified Net access point.
        :type       add_route_table_ids: ``list`` of ``str``

        :param      net_access_point_id: The ID of the Net access point.
        (required)
        :type       net_access_point_id: ``str``

        :param      remove_route_table_ids: One or more IDs of route tables to
        disassociate from the specified Net access point.
        :type       remove_route_table_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The modified Net Access Point
        :rtype: ``dict``
        """
    action = 'UpdateNetAccessPoint'
    data = {'DryRun': dry_run}
    if add_route_table_ids is not None:
        data.update({'AddRouteTablesIds': add_route_table_ids})
    if net_access_point_id is not None:
        data.update({'NetAccessPointId': net_access_point_id})
    if remove_route_table_ids is not None:
        data.update({'RemoveRouteTableIds': remove_route_table_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetAccessPoint']
    return response.json()