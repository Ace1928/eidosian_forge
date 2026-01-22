from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_ip_static_route(self):
    """set ip static route"""
    if not self.changed:
        return
    version = None
    if self.aftype == 'v4':
        version = 'ipv4unicast'
    else:
        version = 'ipv6unicast'
    self.operate_static_route(version, self.prefix, self.mask, self.nhp_interface, self.next_hop, self.vrf, self.destvrf, self.state)