from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def is_vpc_network(self, network_id):
    """Returns True if network is in VPC."""
    if self._vpc_networks_ids is None:
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id')}
        vpcs = self.query_api('listVPCs', **args)
        self._vpc_networks_ids = []
        if vpcs:
            for vpc in vpcs['vpc']:
                for n in vpc.get('network', []):
                    self._vpc_networks_ids.append(n['id'])
    return network_id in self._vpc_networks_ids