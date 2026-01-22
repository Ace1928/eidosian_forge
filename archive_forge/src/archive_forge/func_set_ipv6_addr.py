from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def set_ipv6_addr(self, ifname, addr, mask):
    """Set interface IPv6 address"""
    if not addr or not mask:
        return
    if self.state == 'present':
        self.updates_cmd.append('interface %s' % ifname)
        if self.intf_info['enableFlag'] == 'false':
            xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'true')
            self.netconf_set_config(xml_str, 'SET_IPV6_ENABLE')
            self.updates_cmd.append('ipv6 enable')
            self.changed = True
        if not self.is_ipv6_exist(addr, mask):
            xml_str = CE_NC_ADD_IPV6 % (ifname, addr, mask)
            self.netconf_set_config(xml_str, 'ADD_IPV6_ADDR')
            self.updates_cmd.append('ipv6 address %s %s' % (addr, mask))
            self.changed = True
        if not self.changed:
            self.updates_cmd.pop()
    elif self.is_ipv6_exist(addr, mask):
        xml_str = CE_NC_DEL_IPV6 % (ifname, addr, mask)
        self.netconf_set_config(xml_str, 'DEL_IPV6_ADDR')
        self.updates_cmd.append('interface %s' % ifname)
        self.updates_cmd.append('undo ipv6 address %s %s' % (addr, mask))
        self.changed = True