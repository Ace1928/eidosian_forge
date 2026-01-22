import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_link_internet_service(self, internet_service_id: str=None, net_id: str=None, dry_run: bool=False):
    """
        Attaches an Internet service to a Net.
        To enable the connection between the Internet and a Net, you must
        attach an Internet service to this Net.

        :param      internet_service_id: The ID of the Internet service you
        want to attach. (required)
        :type       internet_service_id: ``str``

        :param      net_id: The ID of the Net to which you want to attach the
        Internet service. (required)
        :type       net_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'LinkInternetService'
    data = {'DryRun': dry_run}
    if internet_service_id is not None:
        data.update({'InternetServiceId': internet_service_id})
    if net_id is not None:
        data.update({'NetId': net_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()