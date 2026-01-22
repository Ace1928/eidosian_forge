from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_vlans_diff(self, vlans, cfg_vlans):
    """Returns the vlan ranges present only in the want vlans list
        and not in the have vlans list
        """
    vlans = self.get_vlan_id_list(vlans)
    cfg_vlans = self.get_vlan_id_list(cfg_vlans)
    return self.get_vlan_range_list(list(set(vlans) - set(cfg_vlans)))