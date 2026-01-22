from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import execute_nc_action, ce_argument_spec
def init_network_module(self, **kwargs):
    """ init network module """
    self.network_module = AnsibleModule(**kwargs)