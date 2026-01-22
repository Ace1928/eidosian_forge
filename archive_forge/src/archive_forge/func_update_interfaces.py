from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.stp.stp import StpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_interfaces(self, data):
    interfaces_list = []
    interfaces = data.get('interfaces', None)
    if interfaces:
        intf_list = interfaces.get('interface', None)
        if intf_list:
            for intf in intf_list:
                intf_dict = {}
                config = intf.get('config', None)
                intf_name = config.get('name', None)
                edge_port = config.get('edge-port', None)
                link_type = config.get('link-type', None)
                guard = config.get('guard', None)
                bpdu_guard = config.get('bpdu-guard', None)
                bpdu_filter = config.get('bpdu-filter', None)
                portfast = config.get('openconfig-spanning-tree-ext:portfast', None)
                uplink_fast = config.get('openconfig-spanning-tree-ext:uplink-fast', None)
                shutdown = config.get('openconfig-spanning-tree-ext:bpdu-guard-port-shutdown', None)
                cost = config.get('openconfig-spanning-tree-ext:cost', None)
                port_priority = config.get('openconfig-spanning-tree-ext:port-priority', None)
                stp_enable = config.get('openconfig-spanning-tree-ext:spanning-tree-enable', None)
                if intf_name:
                    intf_dict['intf_name'] = intf_name
                if edge_port is not None:
                    intf_dict['edge_port'] = stp_map[edge_port]
                if link_type:
                    intf_dict['link_type'] = stp_map[link_type]
                if guard:
                    intf_dict['guard'] = stp_map[guard]
                if bpdu_guard is not None:
                    intf_dict['bpdu_guard'] = bpdu_guard
                if bpdu_filter is not None:
                    intf_dict['bpdu_filter'] = bpdu_filter
                if portfast is not None:
                    intf_dict['portfast'] = portfast
                if uplink_fast is not None:
                    intf_dict['uplink_fast'] = uplink_fast
                if shutdown is not None:
                    intf_dict['shutdown'] = shutdown
                if cost:
                    intf_dict['cost'] = cost
                if port_priority:
                    intf_dict['port_priority'] = port_priority
                if stp_enable is not None:
                    intf_dict['stp_enable'] = stp_enable
                if intf_dict:
                    interfaces_list.append(intf_dict)
    return interfaces_list