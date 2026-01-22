import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_server_certificates(self, paths: str=None, dry_run: bool=False):
    """
        List your server certificates.

        :param      paths: The path to the server certificate.
        :type       paths: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: server certificate
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadServerCertificates'
    data = {'DryRun': dry_run, 'Filters': {}}
    if paths is not None:
        data['Filters'].update({'Paths': paths})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ServerCertificates']
    return response.json()