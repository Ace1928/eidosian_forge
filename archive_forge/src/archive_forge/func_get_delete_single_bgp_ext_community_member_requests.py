from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def get_delete_single_bgp_ext_community_member_requests(self, name, members):
    requests = []
    for member in members:
        url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:'
        url = url + 'bgp-defined-sets/ext-community-sets/ext-community-set={name}/config/{members_param}'
        method = 'DELETE'
        members_params = {'ext-community-member': member}
        members_str = urlencode(members_params)
        request = {'path': url.format(name=name, members_param=members_str), 'method': method}
        requests.append(request)
    return requests