import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_listener_rules(self, listener_rule_names: List[str]=None, dry_run: bool=False):
    """
        Describes one or more listener rules. By default, this action returns
        the full list of listener rules for the account.

        :param      listener_rule_names: The names of the listener rules.
        :type       listener_rule_names: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns the list of Listener Rules
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadListenerRules'
    data = {'DryRun': dry_run, 'Filters': {}}
    if listener_rule_names is not None:
        data['Filters'].update({'ListenerRuleNames': listener_rule_names})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ListenerRules']
    return response.json()