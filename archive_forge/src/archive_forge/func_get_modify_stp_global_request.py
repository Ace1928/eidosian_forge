from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_global_request(self, commands, have):
    request = None
    if not commands:
        return request
    stp_global = commands.get('global', None)
    if stp_global:
        global_dict = {}
        config_dict = {}
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
        if enabled_protocol:
            config_dict['enabled-protocol'] = [stp_map[enabled_protocol]]
        if loop_guard is not None:
            config_dict['loop-guard'] = loop_guard
        if bpdu_filter is not None:
            config_dict['bpdu-filter'] = bpdu_filter
        if disabled_vlans:
            if have:
                cfg_stp_global = have.get('global', None)
                if cfg_stp_global:
                    cfg_disabled_vlans = cfg_stp_global.get('disabled_vlans', None)
                    if cfg_disabled_vlans:
                        disabled_vlans = self.get_vlans_diff(disabled_vlans, cfg_disabled_vlans)
                        if not disabled_vlans:
                            commands['global'].pop('disabled_vlans')
            if disabled_vlans:
                config_dict['openconfig-spanning-tree-ext:disabled-vlans'] = self.convert_vlans_list(disabled_vlans)
        if root_guard_timeout:
            config_dict['openconfig-spanning-tree-ext:rootguard-timeout'] = root_guard_timeout
        if portfast is not None and enabled_protocol == 'pvst':
            config_dict['openconfig-spanning-tree-ext:portfast'] = portfast
        elif portfast:
            self._module.fail_json(msg='Portfast only configurable for pvst protocol.')
        if hello_time:
            config_dict['openconfig-spanning-tree-ext:hello-time'] = hello_time
        if max_age:
            config_dict['openconfig-spanning-tree-ext:max-age'] = max_age
        if fwd_delay:
            config_dict['openconfig-spanning-tree-ext:forwarding-delay'] = fwd_delay
        if bridge_priority:
            config_dict['openconfig-spanning-tree-ext:bridge-priority'] = bridge_priority
        if config_dict:
            global_dict['config'] = config_dict
            url = '%s/global' % STP_PATH
            payload = {'openconfig-spanning-tree:global': global_dict}
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request