from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_all_bfd_cfg_requests(self, commands):
    requests = []
    profiles = commands.get('profiles', None)
    single_hops = commands.get('single_hops', None)
    multi_hops = commands.get('multi_hops', None)
    if profiles:
        url = '%s/openconfig-bfd-ext:bfd-profile/profile' % BFD_PATH
        requests.append({'path': url, 'method': DELETE})
    if single_hops:
        url = '%s/openconfig-bfd-ext:bfd-shop-sessions/single-hop' % BFD_PATH
        requests.append({'path': url, 'method': DELETE})
    if multi_hops:
        url = '%s/openconfig-bfd-ext:bfd-mhop-sessions/multi-hop' % BFD_PATH
        requests.append({'path': url, 'method': DELETE})
    return requests