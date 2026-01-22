import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_listener_rule(self, listener_rule_name: str=None, dry_run: bool=False):
    """
        Deletes a listener rule.
        The previously active rule is disabled after deletion.

        :param      listener_rule_name: The name of the rule you want to
        delete. (required)
        :type       listener_rule_name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteListenerRule'
    data = {'DryRun': dry_run}
    if listener_rule_name is not None:
        data.update({'ListenerRuleName': listener_rule_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()