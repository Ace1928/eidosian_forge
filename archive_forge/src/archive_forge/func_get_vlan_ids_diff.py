from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_vlan_ids_diff(self, vlan_ids, have_vlan_ids, same):
    """ Takes two vlan id lists and finds the difference.
        :param vlan_ids: list of vlan ids that is looking for diffs
        :param have_vlan_ids: list of vlan ids that is being compared to
        :param same: if true will instead return list of shared values
        :rtype: list(str)
        """
    results = []
    for vlan_id in vlan_ids:
        if same:
            if vlan_id in have_vlan_ids:
                results.append(vlan_id)
        elif vlan_id not in have_vlan_ids:
            results.append(vlan_id)
    return results