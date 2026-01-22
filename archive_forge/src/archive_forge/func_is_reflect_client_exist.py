from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def is_reflect_client_exist(self):
    """Judge whether reflect client is configured"""
    view_cmd = 'peer %s reflect-client' % self.peer
    return is_config_exist(self.bgp_evpn_config, view_cmd)