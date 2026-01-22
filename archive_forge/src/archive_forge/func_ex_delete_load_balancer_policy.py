import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_load_balancer_policy(self, load_balancer_name: str=None, policy_name: str=None, dry_run: bool=False):
    """
        Deletes a specified policy from a load balancer.
        In order to be deleted, the policy must not be enabled for any
        listener.

        :param      load_balancer_name: The name of the load balancer for
        which you want to delete a policy. (required)
        :type       load_balancer_name: ``str``

        :param      policy_name: The name of the policy you want to delete.
        (required)
        :type       policy_name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteLoadBalancerPolicy'
    data = {'DryRun': dry_run, 'Tags': {}}
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    if policy_name is not None:
        data['Tags'].update({'PolicyName': policy_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()