from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def preprocess_delete_commands(self, commands, have):
    new_commands = dict()
    if 'ipv4_arp_timeout' in commands:
        new_commands['ipv4_arp_timeout'] = have['ipv4_arp_timeout']
    if 'ipv4_drop_neighbor_aging_time' in commands:
        new_commands['ipv4_drop_neighbor_aging_time'] = have['ipv4_drop_neighbor_aging_time']
    if 'ipv6_drop_neighbor_aging_time' in commands:
        new_commands['ipv6_drop_neighbor_aging_time'] = have['ipv6_drop_neighbor_aging_time']
    if 'ipv6_nd_cache_expiry' in commands:
        new_commands['ipv6_nd_cache_expiry'] = have['ipv6_nd_cache_expiry']
    if 'num_local_neigh' in commands:
        new_commands['num_local_neigh'] = have['num_local_neigh']
    return new_commands