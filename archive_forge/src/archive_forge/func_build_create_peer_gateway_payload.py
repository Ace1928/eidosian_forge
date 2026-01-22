from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def build_create_peer_gateway_payload(self, commands):
    payload = {'openconfig-mclag:vlan-if': []}
    vlan_id_list = self.get_vlan_id_list(commands)
    for vlan in vlan_id_list:
        vlan_name = 'Vlan{0}'.format(vlan)
        payload['openconfig-mclag:vlan-if'].append({'name': vlan_name, 'config': {'name': vlan_name, 'peer-gateway-enable': 'ENABLE'}})
    return payload