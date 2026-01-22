import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_certificate_authority(self, ca_id: str, description: str=None, dry_run: bool=False):
    """
        Modifies the specified attribute of a Client Certificate Authority
        (CA).

        :param      ca_id: The ID of the CA. (required)
        :type       ca_id: ``str``

        :param      description: The description of the CA.
        :type       description: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a the created Ca or the request result.
        :rtype: ``dict``
        """
    action = 'UpdateCa'
    data = {'DryRun': dry_run, 'CaId': ca_id}
    if description is not None:
        data.update({'Description': description})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Ca']
    return response.json()