import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_load_balancer_listeners(self, load_balancer_name: str=None, l_backend_port: int=None, l_backend_protocol: str=None, l_load_balancer_port: int=None, l_load_balancer_protocol: str=None, l_server_certificate_id: str=None, dry_run: bool=False):
    """
        Creates one or more listeners for a specified load balancer.

        :param      load_balancer_name: The name of the load balancer for
        which you want to create listeners. (required)
        :type       load_balancer_name: ``str``

        :param      l_backend_port: The port on which the back-end VM is
        listening (between 1 and 65535, both included). (required)
        :type       l_backend_port: ``int``

        :param      l_backend_protocol: The protocol for routing traffic to
        back-end VMs (HTTP | HTTPS | TCP | SSL | UDP).
        :type       l_backend_protocol: ``int``

        :param      l_load_balancer_port: The port on which the load balancer
        is listening (between 1 and 65535, both included). (required)
        :type       l_load_balancer_port: ``int``

        :param      l_load_balancer_protocol: The routing protocol
        (HTTP | HTTPS | TCP | SSL | UDP). (required)
        :type       l_load_balancer_protocol: ``str``

        :param      l_server_certificate_id: The ID of the server certificate.
        (required)
        :type       l_server_certificate_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Load Balancer Listener
        :rtype: ``dict``
        """
    action = 'CreateLoadBalancerListeners'
    data = {'DryRun': dry_run, 'Listeners': {}}
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    if l_backend_port is not None:
        data['Listeners'].update({'BackendPort': l_backend_port})
    if l_backend_protocol is not None:
        data['Listeners'].update({'BackendProtocol': l_backend_protocol})
    if l_load_balancer_port is not None:
        data['Listeners'].update({'LoadBalancerPort': l_load_balancer_port})
    if l_load_balancer_protocol is not None:
        data['Listeners'].update({'LoadBalancerProtocol': l_load_balancer_protocol})
    if l_server_certificate_id is not None:
        data['Listeners'].update({'ServerCertificateId': l_server_certificate_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LoadBalancer']
    return response.json()