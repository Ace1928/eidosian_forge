import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_read_account(self, dry_run: bool=False):
    """
        Gets information about the account that sent the request.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the account information
        :rtype: ``dict``
        """
    action = 'ReadAccounts'
    data = json.dumps({'DryRun': dry_run})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return response.json()['Accounts'][0]
    return response.json()