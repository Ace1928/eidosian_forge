from __future__ import absolute_import, division, print_function
import json
from copy import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def get_delete_lag_interfaces_requests(self, commands):
    requests = []
    url = 'data/openconfig-interfaces:interfaces/interface={}/openconfig-if-ethernet:ethernet/config/openconfig-if-aggregate:aggregate-id'
    method = DELETE
    for c in commands:
        if c.get('members') and c['members'].get('interfaces'):
            interfaces = c['members']['interfaces']
        else:
            continue
        for each in interfaces:
            ifname = each['member']
            request = {'path': url.format(ifname), 'method': method}
            requests.append(request)
    return requests