from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def get_config_in_bgp_evpn_view(self):
    """Get configuration in BGP_EVPN view"""
    self.bgp_evpn_config = ''
    if not self.config:
        return ''
    index = self.config.find('l2vpn-family evpn')
    if index == -1:
        return ''
    return self.config[index:]