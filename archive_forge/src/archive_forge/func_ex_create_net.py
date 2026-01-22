import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_net(self, ip_range: str=None, tenancy: str=None, dry_run: bool=False):
    """
        Creates a Net with a specified IP range.
        The IP range (network range) of your Net must be between a /28 netmask
        (16 IP addresses) and a /16 netmask (65 536 IP addresses).

        :param      ip_range: The IP range for the Net, in CIDR notation
        (for example, 10.0.0.0/16). (required)
        :type       ip_range: ``str``

        :param      tenancy: The tenancy options for the VMs (default if a VM
        created in a Net can be launched with any tenancy, dedicated if it can
        be launched with dedicated tenancy VMs running on single-tenant
        hardware).
        :type       tenancy: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Nat Service
        :rtype: ``dict``
        """
    action = 'CreateNet'
    data = {'DryRun': dry_run}
    if ip_range is not None:
        data.update({'IpRange': ip_range})
    if tenancy is not None:
        data.update({'Tenancy': tenancy})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Net']
    return response.json()