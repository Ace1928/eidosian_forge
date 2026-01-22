from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def build_host_inbound_traffic(node, host_inbound_traffic):
    host_inbound_traffic_node = build_child_xml_node(node, 'host-inbound-traffic')
    if 'protocols' in host_inbound_traffic:
        for protocol in host_inbound_traffic['protocols']:
            protocol_node = build_child_xml_node(host_inbound_traffic_node, 'protocols')
            build_child_xml_node(protocol_node, 'name', protocol['name'])
            if 'except' in protocol:
                build_child_xml_node(protocol_node, 'except')
    if 'system_services' in host_inbound_traffic:
        for system_service in host_inbound_traffic['system_services']:
            system_service_node = build_child_xml_node(host_inbound_traffic_node, 'system-services')
            build_child_xml_node(system_service_node, 'name', system_service['name'])
            if 'except' in system_service:
                build_child_xml_node(system_service_node, 'except')