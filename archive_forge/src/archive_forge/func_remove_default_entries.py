from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def remove_default_entries(self, data):
    profiles = data.get('profiles', None)
    single_hops = data.get('single_hops', None)
    multi_hops = data.get('multi_hops', None)
    if profiles:
        for profile in profiles:
            enabled = profile.get('enabled', None)
            transmit_interval = profile.get('transmit_interval', None)
            receive_interval = profile.get('receive_interval', None)
            detect_multiplier = profile.get('detect_multiplier', None)
            passive_mode = profile.get('passive_mode', None)
            min_ttl = profile.get('min_ttl', None)
            echo_interval = profile.get('echo_interval', None)
            echo_mode = profile.get('echo_mode', None)
            if enabled:
                profile.pop('enabled')
            if transmit_interval == 300:
                profile.pop('transmit_interval')
            if receive_interval == 300:
                profile.pop('receive_interval')
            if detect_multiplier == 3:
                profile.pop('detect_multiplier')
            if passive_mode is False:
                profile.pop('passive_mode')
            if min_ttl == 254:
                profile.pop('min_ttl')
            if echo_interval == 300:
                profile.pop('echo_interval')
            if echo_mode is False:
                profile.pop('echo_mode')
    if single_hops:
        for hop in single_hops:
            enabled = hop.get('enabled', None)
            transmit_interval = hop.get('transmit_interval', None)
            receive_interval = hop.get('receive_interval', None)
            detect_multiplier = hop.get('detect_multiplier', None)
            passive_mode = hop.get('passive_mode', None)
            echo_interval = hop.get('echo_interval', None)
            echo_mode = hop.get('echo_mode', None)
            if enabled:
                hop.pop('enabled')
            if transmit_interval == 300:
                hop.pop('transmit_interval')
            if receive_interval == 300:
                hop.pop('receive_interval')
            if detect_multiplier == 3:
                hop.pop('detect_multiplier')
            if passive_mode is False:
                hop.pop('passive_mode')
            if echo_interval == 300:
                hop.pop('echo_interval')
            if echo_mode is False:
                hop.pop('echo_mode')
    if multi_hops:
        for hop in multi_hops:
            enabled = hop.get('enabled', None)
            transmit_interval = hop.get('transmit_interval', None)
            receive_interval = hop.get('receive_interval', None)
            detect_multiplier = hop.get('detect_multiplier', None)
            passive_mode = hop.get('passive_mode', None)
            min_ttl = hop.get('min_ttl', None)
            if enabled:
                hop.pop('enabled')
            if transmit_interval == 300:
                hop.pop('transmit_interval')
            if receive_interval == 300:
                hop.pop('receive_interval')
            if detect_multiplier == 3:
                hop.pop('detect_multiplier')
            if passive_mode is False:
                hop.pop('passive_mode')
            if min_ttl == 254:
                hop.pop('min_ttl')