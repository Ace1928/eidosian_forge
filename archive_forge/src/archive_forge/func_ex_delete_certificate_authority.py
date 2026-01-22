import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_certificate_authority(self, ca_id: str, dry_run: bool=False):
    """
        Deletes a specified Client Certificate Authority (CA).

        :param      ca_id: The ID of the CA you want to delete. (required)
        :type       ca_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``
        """
    action = 'DeleteCa'
    data = {'DryRun': dry_run, 'CaId': ca_id}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()