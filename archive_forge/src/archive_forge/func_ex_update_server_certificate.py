import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_server_certificate(self, name: str=None, new_name: str=None, new_path: str=None, dry_run: bool=False):
    """
        Modifies the name and/or the path of a specified server certificate.

        :param      name: The name of the server certificate
        you want to modify.
        :type       name: ``str``

        :param      new_name: A new name for the server certificate.
        :type       new_name: ``str``

        :param      new_path:A new path for the server certificate.
        :type       new_path: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the new server certificate
        :rtype: ``dict``
        """
    action = 'UpdateServerCertificate'
    data = {'DryRun': dry_run}
    if name is not None:
        data.update({'Name': name})
    if new_name is not None:
        data.update({'NewName': new_name})
    if new_path is not None:
        data.update({'NewPath': new_path})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ServerCertificate']
    return response.json()