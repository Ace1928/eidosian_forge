from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_mclag_domain_requests(self, have):
    requests = []
    path = 'data/openconfig-mclag:mclag/mclag-domains'
    method = DELETE
    if have.get('peer_gateway'):
        request = {'path': 'data/openconfig-mclag:mclag/vlan-ifs/vlan-if', 'method': method}
        requests.append(request)
    if have.get('unique_ip'):
        request = {'path': 'data/openconfig-mclag:mclag/vlan-interfaces/vlan-interface', 'method': method}
        requests.append(request)
    if have.get('gateway_mac'):
        request = {'path': 'data/openconfig-mclag:mclag/mclag-gateway-macs/mclag-gateway-mac', 'method': method}
        requests.append(request)
    request = {'path': path, 'method': method}
    requests.append(request)
    return requests