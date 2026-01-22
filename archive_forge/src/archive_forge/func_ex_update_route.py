import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_route(self, destination_ip_range: str=None, gateway_id: str=None, nat_service_id: str=None, net_peering_id: str=None, nic_id: str=None, route_table_id: str=None, vm_id: str=None, dry_run: bool=False):
    """
        Replaces an existing route within a route table in a Net.
        You must specify one of the following elements as the target:

        - Net peering connection
        - NAT virtual machine (VM)
        - Internet service
        - Virtual gateway
        - NAT service
        - Network interface card (NIC)

        The routing algorithm is based on the most specific match.

        :param      destination_ip_range: The IP range used for the
        destination match, in CIDR notation (for example, 10.0.0.0/24).
        (required)
        :type       destination_ip_range: ``str``

        :param      gateway_id: The ID of an Internet service or virtual
        gateway attached to your Net.
        :type       gateway_id: ``str``

        :param      nat_service_id: The ID of a NAT service.
        :type       nat_service_id: ``str``

        :param      net_peering_id: The ID of a Net peering connection.
        :type       net_peering_id: ``str``

        :param      nic_id: The ID of a NIC.
        :type       nic_id: ``str``

        :param      vm_id: The ID of a NAT VM in your Net (attached to exactly
        one NIC).
        :type       vm_id: ``str``

        :param      route_table_id: The ID of the route table for which you
        want to create a route. (required)
        :type       route_table_id: `str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The updated Route
        :rtype: ``dict``
        """
    action = 'UpdateRoute'
    data = {'DryRun': dry_run}
    if destination_ip_range is not None:
        data.update({'DestinationIpRange': destination_ip_range})
    if gateway_id is not None:
        data.update({'GatewayId': gateway_id})
    if nat_service_id is not None:
        data.update({'NatServiceId': nat_service_id})
    if net_peering_id is not None:
        data.update({'NetPeeringId': net_peering_id})
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if route_table_id is not None:
        data.update({'RouteTableId': route_table_id})
    if vm_id is not None:
        data.update({'VmId': vm_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['RouteTable']
    return response.json()