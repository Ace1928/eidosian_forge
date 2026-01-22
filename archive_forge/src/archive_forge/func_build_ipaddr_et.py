from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def build_ipaddr_et(self, config, unit_node, protocol='ipv4', delete=False):
    family = build_child_xml_node(unit_node, 'family')
    inet = 'inet'
    if protocol == 'ipv6':
        inet = 'inet6'
    ip_protocol = build_child_xml_node(family, inet)
    for ip_addr in config[protocol]:
        if ip_addr['address'] == 'dhcp' and protocol == 'ipv4':
            build_child_xml_node(ip_protocol, 'dhcp')
        else:
            ip_addresses = build_child_xml_node(ip_protocol, 'address')
            build_child_xml_node(ip_addresses, 'name', ip_addr['address'])