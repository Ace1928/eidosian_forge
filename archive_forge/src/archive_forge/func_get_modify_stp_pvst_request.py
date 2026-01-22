from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_pvst_request(self, commands):
    request = None
    pvst = commands.get('pvst', None)
    if pvst:
        vlans_list = self.get_vlans_list(pvst)
        if vlans_list:
            url = '%s/openconfig-spanning-tree-ext:pvst' % STP_PATH
            payload = {'openconfig-spanning-tree-ext:pvst': {'vlans': vlans_list}}
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request