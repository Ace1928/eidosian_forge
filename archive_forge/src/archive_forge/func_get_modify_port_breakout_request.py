from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_modify_port_breakout_request(self, conf, match):
    request = None
    name = conf.get('name', None)
    mode = conf.get('mode', None)
    url = 'data/openconfig-platform:components'
    payload = self.get_port_breakout_payload(name, mode, match)
    request = {'path': url, 'method': PATCH, 'data': payload}
    return request