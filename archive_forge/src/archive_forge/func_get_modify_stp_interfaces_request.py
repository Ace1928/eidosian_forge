from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_interfaces_request(self, commands):
    request = None
    interfaces = commands.get('interfaces', None)
    if interfaces:
        intf_list = []
        for intf in interfaces:
            intf_dict = {}
            config_dict = {}
            intf_name = intf.get('intf_name', None)
            edge_port = intf.get('edge_port', None)
            link_type = intf.get('link_type', None)
            guard = intf.get('guard', None)
            bpdu_guard = intf.get('bpdu_guard', None)
            bpdu_filter = intf.get('bpdu_filter', None)
            portfast = intf.get('portfast', None)
            uplink_fast = intf.get('uplink_fast', None)
            shutdown = intf.get('shutdown', None)
            cost = intf.get('cost', None)
            port_priority = intf.get('port_priority', None)
            stp_enable = intf.get('stp_enable', None)
            if intf_name:
                config_dict['name'] = intf_name
            if edge_port is not None:
                config_dict['edge-port'] = stp_map[edge_port]
            if link_type:
                config_dict['link-type'] = stp_map[link_type]
            if guard:
                config_dict['guard'] = stp_map[guard]
            if bpdu_guard is not None:
                config_dict['bpdu-guard'] = bpdu_guard
            if bpdu_filter is not None:
                config_dict['bpdu-filter'] = bpdu_filter
            if portfast is not None:
                config_dict['openconfig-spanning-tree-ext:portfast'] = portfast
            if uplink_fast is not None:
                config_dict['openconfig-spanning-tree-ext:uplink-fast'] = uplink_fast
            if shutdown is not None:
                config_dict['openconfig-spanning-tree-ext:bpdu-guard-port-shutdown'] = shutdown
            if cost:
                config_dict['openconfig-spanning-tree-ext:cost'] = cost
            if port_priority:
                config_dict['openconfig-spanning-tree-ext:port-priority'] = port_priority
            if stp_enable is not None:
                config_dict['openconfig-spanning-tree-ext:spanning-tree-enable'] = stp_enable
            if config_dict:
                intf_dict['name'] = intf_name
                intf_dict['config'] = config_dict
                intf_list.append(intf_dict)
        if intf_list:
            url = '%s/interfaces' % STP_PATH
            payload = {'openconfig-spanning-tree:interfaces': {'interface': intf_list}}
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request