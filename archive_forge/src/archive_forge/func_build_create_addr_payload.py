from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def build_create_addr_payload(self, ip, mask, secondary=None):
    cfg = {'ip': ip, 'prefix-length': float(mask)}
    if secondary:
        cfg['secondary'] = secondary
    addr_payload = {'ip': ip, 'openconfig-if-ip:config': cfg}
    return addr_payload