from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_vlans_requests(self, configs):
    requests = []
    if not configs:
        return requests
    for vlan in configs:
        vlan_id = vlan.get('vlan_id')
        interface_name = 'Vlan' + str(vlan_id)
        description = vlan.get('description', None)
        request = build_interfaces_create_request(interface_name=interface_name)
        requests.append(request)
        if description:
            requests.append(self.get_modify_vlan_config_attr(interface_name, 'description', description))
    return requests