from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_radius_global_ext_payload(self, conf):
    payload = {}
    global_ext_cfg = {}
    if conf.get('nas_ip', None):
        global_ext_cfg['nas-ip-address'] = conf['nas_ip']
    if conf.get('retransmit', None):
        global_ext_cfg['retransmit-attempts'] = conf['retransmit']
    if conf.get('statistics', None):
        global_ext_cfg['statistics'] = conf['statistics']
    if global_ext_cfg:
        payload = {'openconfig-aaa-radius-ext:config': global_ext_cfg}
    return payload