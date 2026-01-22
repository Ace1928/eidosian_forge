from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def find_nei(self, have, bgp_as, vrf_name, neighbor):
    mat_dict = next((m_neighbor for m_neighbor in have if m_neighbor['bgp_as'] == bgp_as and m_neighbor['vrf_name'] == vrf_name), None)
    if mat_dict and mat_dict.get('neighbors', None) is not None:
        mat_neighbor = next((m for m in mat_dict['neighbors'] if m['neighbor'] == neighbor['neighbor']), None)
        return mat_neighbor