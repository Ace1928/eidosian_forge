import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_secret_access_key(self, access_key_id: str=None, dry_run: bool=False):
    """
        Gets information about the secret access key associated with
        the account that sends the request.

        :param      access_key_id: The ID of the access key. (required)
        :type       access_key_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Access Key
        :rtype: ``dict``
        """
    action = 'ReadSecretAccessKey'
    data = {'DryRun': dry_run}
    if access_key_id is not None:
        data.update({'AccessKeyId': access_key_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['AccessKey']
    return response.json()