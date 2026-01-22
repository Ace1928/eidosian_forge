from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
@staticmethod
def get_vlan_range_list(vlan_id_list):
    """Returns a list of VLAN ranges for given list of VLAN IDs
        in vlans spec format"""
    vlan_range_list = []
    if vlan_id_list:
        vlan_id_list.sort()
        for vlan_range in get_ranges_in_list(vlan_id_list):
            vlan_range_list.append({'vlan': 'Vlan{0}'.format('-'.join(map(str, (vlan_range[0], vlan_range[-1])[:len(vlan_range)])))})
    return vlan_range_list