import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_listener_rule(self, host_pattern: str=None, listener_rule_name: str=None, path_pattern: str=None, dry_run: bool=False):
    """
        Updates the pattern of the listener rule.
        This call updates the pattern matching algorithm for incoming traffic.

        :param      host_pattern: TA host-name pattern for the rule, with a
        maximum length of 128 characters. This host-name pattern supports
        maximum three wildcards, and must not contain any special characters
        except [-.?].
        :type       host_pattern: ``str`

        :param      listener_rule_name: The name of the listener rule.
        (required)
        :type       listener_rule_name: ``str``

        :param      path_pattern: A path pattern for the rule, with a maximum
        length of 128 characters. This path pattern supports maximum three
        wildcards, and must not contain any special characters
        except [_-.$/~"'@:+?].
        :type       path_pattern: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Update the specified Listener Rule
        :rtype: ``dict``
        """
    action = 'UpdateListenerRule'
    data = {'DryRun': dry_run}
    if host_pattern is not None:
        data.update({'HostPattern': host_pattern})
    if listener_rule_name is not None:
        data.update({'ListenerRuleName': listener_rule_name})
    if path_pattern is not None:
        data.update({'PathPattern': path_pattern})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ListenerRule']
    return response.json()