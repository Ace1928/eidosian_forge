from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bfd.bfd import BfdArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_multi_hops(self, data):
    all_multi_hops = []
    bfd_multi_hop = data.get('openconfig-bfd-ext:bfd-mhop-sessions', None)
    if bfd_multi_hop:
        multi_hop_list = bfd_multi_hop.get('multi-hop', None)
        if multi_hop_list:
            for hop in multi_hop_list:
                multi_hop_dict = {}
                remote_address = hop['remote-address']
                vrf = hop['vrf']
                local_address = hop['local-address']
                config = hop['config']
                enabled = config.get('enabled', None)
                transmit_interval = config.get('desired-minimum-tx-interval', None)
                receive_interval = config.get('required-minimum-receive', None)
                detect_multiplier = config.get('detection-multiplier', None)
                passive_mode = config.get('passive-mode', None)
                min_ttl = config.get('minimum-ttl', None)
                profile_name = config.get('profile-name', None)
                if remote_address:
                    multi_hop_dict['remote_address'] = remote_address
                if vrf:
                    multi_hop_dict['vrf'] = vrf
                if local_address:
                    multi_hop_dict['local_address'] = local_address
                if enabled is not None:
                    multi_hop_dict['enabled'] = enabled
                if transmit_interval:
                    multi_hop_dict['transmit_interval'] = transmit_interval
                if receive_interval:
                    multi_hop_dict['receive_interval'] = receive_interval
                if detect_multiplier:
                    multi_hop_dict['detect_multiplier'] = detect_multiplier
                if passive_mode is not None:
                    multi_hop_dict['passive_mode'] = passive_mode
                if min_ttl:
                    multi_hop_dict['min_ttl'] = min_ttl
                if profile_name:
                    multi_hop_dict['profile_name'] = profile_name
                if multi_hop_dict:
                    all_multi_hops.append(multi_hop_dict)
    return all_multi_hops