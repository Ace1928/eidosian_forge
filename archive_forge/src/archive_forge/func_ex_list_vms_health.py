import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_vms_health(self, backend_vm_ids: List[str]=None, load_balancer_name: str=None, dry_run: bool=False):
    """
        Lists the state of one or more back-end virtual machines (VMs)
        registered with a specified load balancer.

        :param      load_balancer_name: The name of the load balancer.
        (required)
        :type       load_balancer_name: ``str``

        :param      backend_vm_ids: One or more IDs of back-end VMs.
        :type       backend_vm_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of back end vms health
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadVmsHealth'
    data = {'DryRun': dry_run}
    if backend_vm_ids is not None:
        data.update({'BackendVmIds': backend_vm_ids})
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['BackendVmHealth']
    return response.json()