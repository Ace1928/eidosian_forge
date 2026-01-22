from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bfd.bfd import BfdArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_single_hops(self, data):
    all_single_hops = []
    bfd_single_hop = data.get('openconfig-bfd-ext:bfd-shop-sessions', None)
    if bfd_single_hop:
        single_hop_list = bfd_single_hop.get('single-hop', None)
        if single_hop_list:
            for hop in single_hop_list:
                single_hop_dict = {}
                remote_address = hop['remote-address']
                vrf = hop['vrf']
                interface = hop['interface']
                local_address = hop['local-address']
                config = hop['config']
                enabled = config.get('enabled', None)
                transmit_interval = config.get('desired-minimum-tx-interval', None)
                receive_interval = config.get('required-minimum-receive', None)
                detect_multiplier = config.get('detection-multiplier', None)
                passive_mode = config.get('passive-mode', None)
                echo_interval = config.get('desired-minimum-echo-receive', None)
                echo_mode = config.get('echo-active', None)
                profile_name = config.get('profile-name', None)
                if remote_address:
                    single_hop_dict['remote_address'] = remote_address
                if vrf:
                    single_hop_dict['vrf'] = vrf
                if interface:
                    single_hop_dict['interface'] = interface
                if local_address:
                    single_hop_dict['local_address'] = local_address
                if enabled is not None:
                    single_hop_dict['enabled'] = enabled
                if transmit_interval:
                    single_hop_dict['transmit_interval'] = transmit_interval
                if receive_interval:
                    single_hop_dict['receive_interval'] = receive_interval
                if detect_multiplier:
                    single_hop_dict['detect_multiplier'] = detect_multiplier
                if passive_mode is not None:
                    single_hop_dict['passive_mode'] = passive_mode
                if echo_interval:
                    single_hop_dict['echo_interval'] = echo_interval
                if echo_mode is not None:
                    single_hop_dict['echo_mode'] = echo_mode
                if profile_name:
                    single_hop_dict['profile_name'] = profile_name
                if single_hop_dict:
                    all_single_hops.append(single_hop_dict)
    return all_single_hops