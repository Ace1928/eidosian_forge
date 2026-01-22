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
def get_trunk_allowed_vlans_diff(self, config, match):
    """Returns the allowed vlan ranges present only in 'config'
        and and not in 'match' in allowed_vlans spec format
        """
    trunk_vlans = []
    match_trunk_vlans = []
    if config.get('trunk') and config['trunk'].get('allowed_vlans'):
        trunk_vlans = config['trunk']['allowed_vlans']
    if not trunk_vlans:
        return []
    if match.get('trunk') and match['trunk'].get('allowed_vlans'):
        match_trunk_vlans = match['trunk']['allowed_vlans']
    if not match_trunk_vlans:
        return trunk_vlans
    trunk_vlans = self.get_vlan_id_list(trunk_vlans)
    match_trunk_vlans = self.get_vlan_id_list(match_trunk_vlans)
    return self.get_allowed_vlan_range_list(list(set(trunk_vlans) - set(match_trunk_vlans)))