import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_register_vms_in_load_balancer(self, backend_vm_ids: List[str]=None, load_balancer_name: str=None, dry_run: bool=False):
    """
        Registers one or more virtual machines (VMs) with a specified load
        balancer.
        The VMs must be running in the same network as the load balancer
        (in the public Cloud or in the same Net). It may take a little time
        for a VM to be registered with the load balancer. Once the VM is
        registered with a load balancer, it receives traffic and requests from
        this load balancer and is called a back-end VM.

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
    action = 'RegisterVmsInLoadBalancer'
    data = {'DryRun': dry_run, 'Filters': {}}
    if backend_vm_ids is not None:
        data.update({'BackendVmIds': backend_vm_ids})
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['BackendVmHealth']
    return response.json()