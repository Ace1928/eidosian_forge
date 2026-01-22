import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_direct_link(self, bandwidth: str=None, direct_link_name: str=None, location: str=None, dry_run: bool=False):
    """
        Creates a new DirectLink between a customer network and a
        specified DirectLink location.

        :param      bandwidth: The bandwidth of the DirectLink
        (1Gbps | 10Gbps). (required)
        :type       bandwidth: ``str``

        :param      direct_link_name: The name of the DirectLink. (required)
        :type       direct_link_name: ``str``

        :param      location: The code of the requested location for
        the DirectLink, returned by the list_locations method.
        Protocol (NTP) servers.
        :type       location: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Direct Link
        :rtype: ``dict``
        """
    action = 'CreateDirectLink'
    data = {'DryRun': dry_run}
    if bandwidth is not None:
        data.update({'Bandwidth': bandwidth})
    if direct_link_name is not None:
        data.update({'DirectLinkName': direct_link_name})
    if location is not None:
        data.update({'Location': location})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['DirectLink']
    return response.json()