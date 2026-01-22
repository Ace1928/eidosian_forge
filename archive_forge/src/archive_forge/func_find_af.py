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
def find_af(self, have, bgp_as, vrf_name, peergroup, afi, safi):
    mat_pg = self.find_pg(have, bgp_as, vrf_name, peergroup)
    if mat_pg and mat_pg['address_family'].get('afis', None) is not None:
        mat_af = next((af for af in mat_pg['address_family']['afis'] if af['afi'] == afi and af['safi'] == safi), None)
        return mat_af