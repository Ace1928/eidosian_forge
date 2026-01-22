import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_link_private_ips(self, allow_relink: bool=None, nic_id: str=None, private_ips: List[str]=None, secondary_private_ip_count: int=None, dry_run: bool=False):
    """
        Assigns one or more secondary private IP addresses to a specified
        network interface card (NIC). This action is only available in a Net.
        The private IP addresses to be assigned can be added individually
        using the PrivateIps parameter, or you can specify the number of
        private IP addresses to be automatically chosen within the Subnet
        range using the SecondaryPrivateIpCount parameter. You can specify
        only one of these two parameters. If none of these parameters are
        specified, a private IP address is chosen within the Subnet range.

        :param      allow_relink: If true, allows an IP address that is
        already assigned to another NIC in the same Subnet to be assigned
        to the NIC you specified.
        :type       allow_relink: ``str``

        :param      nic_id: The ID of the NIC. (required)
        :type       nic_id: ``str``

        :param      private_ips: The secondary private IP address or addresses
        you want to assign to the NIC within the IP address range of the
        Subnet.
        :type       private_ips: ``list`` of ``str``

        :param      secondary_private_ip_count: The secondary private IP a
        ddress or addresses you want to assign to the NIC within the IP
        address range of the Subnet.
        :type       secondary_private_ip_count: ``int``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return:True if the action is successful
        :rtype: ``bool``
        """
    action = 'LinkPrivateIps'
    data = {'DryRun': dry_run}
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if allow_relink is not None:
        data.update({'AllowRelink': allow_relink})
    if private_ips is not None:
        data.update({'PrivateIps': private_ips})
    if secondary_private_ip_count is not None:
        data.update({'SecondaryPrivateIpCount': secondary_private_ip_count})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()