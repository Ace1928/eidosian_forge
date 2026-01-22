from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_stp_global_requests(self, commands, have):
    requests = []
    stp_global = commands.get('global', None)
    if stp_global:
        enabled_protocol = stp_global.get('enabled_protocol', None)
        loop_guard = stp_global.get('loop_guard', None)
        bpdu_filter = stp_global.get('bpdu_filter', None)
        disabled_vlans = stp_global.get('disabled_vlans', None)
        root_guard_timeout = stp_global.get('root_guard_timeout', None)
        portfast = stp_global.get('portfast', None)
        hello_time = stp_global.get('hello_time', None)
        max_age = stp_global.get('max_age', None)
        fwd_delay = stp_global.get('fwd_delay', None)
        bridge_priority = stp_global.get('bridge_priority', None)
        cfg_stp_global = have.get('global', None)
        if cfg_stp_global:
            cfg_enabled_protocol = cfg_stp_global.get('enabled_protocol', None)
            cfg_loop_guard = cfg_stp_global.get('loop_guard', None)
            cfg_bpdu_filter = cfg_stp_global.get('bpdu_filter', None)
            cfg_disabled_vlans = cfg_stp_global.get('disabled_vlans', None)
            cfg_root_guard_timeout = cfg_stp_global.get('root_guard_timeout', None)
            cfg_portfast = cfg_stp_global.get('portfast', None)
            cfg_hello_time = cfg_stp_global.get('hello_time', None)
            cfg_max_age = cfg_stp_global.get('max_age', None)
            cfg_fwd_delay = cfg_stp_global.get('fwd_delay', None)
            cfg_bridge_priority = cfg_stp_global.get('bridge_priority', None)
            if loop_guard and loop_guard == cfg_loop_guard:
                requests.append(self.get_delete_stp_global_attr('loop-guard'))
            if bpdu_filter and bpdu_filter == cfg_bpdu_filter:
                requests.append(self.get_delete_stp_global_attr('bpdu-filter'))
            if disabled_vlans and cfg_disabled_vlans:
                disabled_vlans_to_delete = self.get_vlans_common(disabled_vlans, cfg_disabled_vlans)
                for i, vlan in enumerate(disabled_vlans_to_delete):
                    if '-' in vlan:
                        disabled_vlans_to_delete[i] = vlan.replace('-', '..')
                if disabled_vlans_to_delete:
                    encoded_vlans = '%2C'.join(disabled_vlans_to_delete)
                    attr = 'openconfig-spanning-tree-ext:disabled-vlans=%s' % encoded_vlans
                    requests.append(self.get_delete_stp_global_attr(attr))
                else:
                    commands['global'].pop('disabled_vlans')
            if root_guard_timeout:
                if root_guard_timeout == cfg_root_guard_timeout:
                    requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:rootguard-timeout'))
                else:
                    commands['global'].pop('root_guard_timeout')
            if portfast and portfast == cfg_portfast:
                requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:portfast'))
            if hello_time and hello_time == cfg_hello_time:
                requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:hello-time'))
            if max_age and max_age == cfg_max_age:
                requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:max-age'))
            if fwd_delay and fwd_delay == cfg_fwd_delay:
                requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:forwarding-delay'))
            if bridge_priority and bridge_priority == cfg_bridge_priority:
                requests.append(self.get_delete_stp_global_attr('openconfig-spanning-tree-ext:bridge-priority'))
            if enabled_protocol:
                if enabled_protocol == cfg_enabled_protocol:
                    requests.append(self.get_delete_stp_global_attr('enabled-protocol'))
                else:
                    commands['global'].pop('enabled_protocol')
    return requests