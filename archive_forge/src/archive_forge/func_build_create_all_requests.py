from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_all_requests(self):
    requests = []
    payload = {'openconfig-neighbor:neighbor-global': [{'name': 'Values', 'config': IP_NEIGH_CONFIG_REQ_DEFAULT}]}
    method = PUT
    request = {'path': GLB_URL, 'method': method, 'data': payload}
    requests.append(request)
    return requests