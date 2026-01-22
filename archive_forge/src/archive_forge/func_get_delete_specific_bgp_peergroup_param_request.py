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
def get_delete_specific_bgp_peergroup_param_request(self, vrf_name, cmd, want_match):
    requests = []
    want_peer_group = want_match.get('peer_group', None)
    for each in cmd['peer_group']:
        if each:
            name = each.get('name', None)
            remote_as = each.get('remote_as', None)
            timers = each.get('timers', None)
            advertisement_interval = each.get('advertisement_interval', None)
            bfd = each.get('bfd', None)
            capability = each.get('capability', None)
            auth_pwd = each.get('auth_pwd', None)
            pg_description = each.get('pg_description', None)
            disable_connected_check = each.get('disable_connected_check', None)
            dont_negotiate_capability = each.get('dont_negotiate_capability', None)
            ebgp_multihop = each.get('ebgp_multihop', None)
            enforce_first_as = each.get('enforce_first_as', None)
            enforce_multihop = each.get('enforce_multihop', None)
            local_address = each.get('local_address', None)
            local_as = each.get('local_as', None)
            override_capability = each.get('override_capability', None)
            passive = each.get('passive', None)
            shutdown_msg = each.get('shutdown_msg', None)
            solo = each.get('solo', None)
            strict_capability_match = each.get('strict_capability_match', None)
            ttl_security = each.get('ttl_security', None)
            address_family = each.get('address_family', None)
            if name and (not remote_as) and (not timers) and (not advertisement_interval) and (not bfd) and (not capability) and (not auth_pwd) and (not pg_description) and (disable_connected_check is None) and (dont_negotiate_capability is None) and (not ebgp_multihop) and (enforce_first_as is None) and (enforce_multihop is None) and (not local_address) and (not local_as) and (override_capability is None) and (passive is None) and (not shutdown_msg) and (solo is None) and (strict_capability_match is None) and (not ttl_security) and (not address_family):
                want_pg_match = None
                if want_peer_group:
                    want_pg_match = next((cfg for cfg in want_peer_group if cfg['name'] == name), None)
                if want_pg_match:
                    keys = ['remote_as', 'timers', 'advertisement_interval', 'bfd', 'capability', 'auth_pwd', 'pg_description', 'disable_connected_check', 'dont_negotiate_capability', 'ebgp_multihop', 'enforce_first_as', 'enforce_multihop', 'local_address', 'local_as', 'override_capability', 'passive', 'shutdown_msg', 'solo', 'strict_capability_match', 'ttl_security', 'address_family']
                    if not any((want_pg_match.get(key, None) for key in keys)):
                        requests.append(self.get_delete_vrf_specific_peergroup_request(vrf_name, name))
            else:
                requests.extend(self.delete_specific_peergroup_param_request(vrf_name, each))
    return requests