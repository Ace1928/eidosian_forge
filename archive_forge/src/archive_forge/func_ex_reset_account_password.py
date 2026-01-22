import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_reset_account_password(self, password: str='', token: str='', dry_run: bool=False):
    """
        Sends an email to the email address provided for the account with a
        token to reset your password.

        :param      password: The new password for the account.
        :type       password: ``str``

        :param      token: The token you received at the email address
        provided for the account.
        :type       token: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'ResetAccountPassword'
    data = json.dumps({'DryRun': dry_run, 'Password': password, 'Token': token})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return True
    return response.json()