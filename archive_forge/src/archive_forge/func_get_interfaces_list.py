from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_interfaces_list(self, interfaces):
    intf_list = []
    for intf in interfaces:
        intf_dict = {}
        intf_cfg_dict = {}
        intf_name = intf.get('intf_name', None)
        cost = intf.get('cost', None)
        port_priority = intf.get('port_priority', None)
        if intf_name:
            intf_cfg_dict['name'] = intf_name
        if cost:
            intf_cfg_dict['cost'] = cost
        if port_priority:
            intf_cfg_dict['port-priority'] = port_priority
        if intf_cfg_dict:
            intf_dict['name'] = intf_name
            intf_dict['config'] = intf_cfg_dict
            intf_list.append(intf_dict)
    return intf_list