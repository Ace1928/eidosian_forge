import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_load_balancer_tags(self, load_balancer_names: List[str]=None, dry_run: bool=False):
    """
        Describes the tags associated with one or more specified load
        balancers.

        :param      load_balancer_names: The names of the load balancer.
        (required)
        :type       load_balancer_names: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of load balancer tags
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadLoadBalancerTags'
    data = {'DryRun': dry_run}
    if load_balancer_names is not None:
        data.update({'LoadBalancerNames': load_balancer_names})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Tags']
    return response.json()