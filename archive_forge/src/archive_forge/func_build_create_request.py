from __future__ import absolute_import, division, print_function
from natsort import (
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import re
import traceback
def build_create_request(self, c_attr, h_attr, intf_name, attr):
    attributes_payload = {'speed': 'port-speed', 'auto_negotiate': 'auto-negotiate', 'fec': 'openconfig-if-ethernet-ext2:port-fec', 'advertised_speed': 'openconfig-if-ethernet-ext2:advertised-speed'}
    config_url = (url + eth_conf_url) % quote(intf_name, safe='')
    payload = {'openconfig-if-ethernet:config': {}}
    payload_attr = attributes_payload.get(attr, attr)
    method = PATCH
    if attr in ('description', 'mtu', 'enabled'):
        config_url = (url + '/config') % quote(intf_name, safe='')
        payload = {'openconfig-interfaces:config': {}}
        payload['openconfig-interfaces:config'][payload_attr] = c_attr
        return {'path': config_url, 'method': method, 'data': payload}
    elif attr in 'fec':
        payload['openconfig-if-ethernet:config'][payload_attr] = 'openconfig-platform-types:' + c_attr
        return {'path': config_url, 'method': method, 'data': payload}
    else:
        payload['openconfig-if-ethernet:config'][payload_attr] = c_attr
        if attr == 'speed':
            if self.is_port_in_port_group(intf_name):
                self._module.fail_json(msg='Unable to configure speed in port group member. Please use port group module to change the speed')
            payload['openconfig-if-ethernet:config'][payload_attr] = 'openconfig-if-ethernet:' + c_attr
        if attr == 'advertised_speed':
            c_ads = c_attr if c_attr else []
            h_ads = h_attr if h_attr else []
            new_ads = list(set(h_ads).union(c_ads))
            if new_ads:
                payload['openconfig-if-ethernet:config'][payload_attr] = ','.join(new_ads)
        return {'path': config_url, 'method': method, 'data': payload}
    return []