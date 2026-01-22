from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def create_loadbalancerpool(self, alias, location, lb_id, method, persistence, port):
    """
        Creates a pool on the provided load balancer
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the load balancer
        :param method: the load balancing method
        :param persistence: the load balancing persistence type
        :param port: the port that the load balancer will listen on
        :return: result: The result from the create API call
        """
    result = None
    try:
        result = self.clc.v2.API.Call('POST', '/v2/sharedLoadBalancers/%s/%s/%s/pools' % (alias, location, lb_id), json.dumps({'port': port, 'method': method, 'persistence': persistence}))
    except APIFailedResponse as e:
        self.module.fail_json(msg='Unable to create pool for load balancer id "{0}". {1}'.format(lb_id, str(e.response_text)))
    return result