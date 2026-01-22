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
def get_interface_requests(self, configs, have):
    requests = []
    if not configs:
        return requests
    for conf in configs:
        name = conf['name']
        have_conf = next((cfg for cfg in have if cfg['name'] == name), None)
        if name.startswith('Loopback'):
            if not have_conf:
                loopback_create_request = build_interfaces_create_request(name)
                requests.append(loopback_create_request)
        else:
            attribute = eth_attribute if name.startswith('Eth') else non_eth_attribute
            for attr in attribute:
                if attr in conf:
                    c_attr = conf.get(attr)
                    h_attr = have_conf.get(attr)
                    attr_request = self.build_create_request(c_attr, h_attr, name, attr)
                    if attr_request:
                        requests.append(attr_request)
    return requests