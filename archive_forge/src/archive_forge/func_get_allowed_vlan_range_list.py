from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
@staticmethod
def get_allowed_vlan_range_list(vlan_id_list):
    """Returns the allowed_vlans list for given list of VLAN IDs"""
    allowed_vlan_range_list = []
    if vlan_id_list:
        vlan_id_list.sort()
        for vlan_range in get_ranges_in_list(vlan_id_list):
            allowed_vlan_range_list.append({'vlan': '-'.join(map(str, (vlan_range[0], vlan_range[-1])[:len(vlan_range)]))})
    return allowed_vlan_range_list