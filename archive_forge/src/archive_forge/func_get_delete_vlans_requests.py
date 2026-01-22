from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.interfaces_util import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vlans_requests(self, configs, delete_vlan=False):
    requests = []
    if not configs:
        return requests
    url = 'data/openconfig-interfaces:interfaces/interface=Vlan{}'
    method = 'DELETE'
    for vlan in configs:
        vlan_id = vlan.get('vlan_id')
        description = vlan.get('description')
        if description and (not delete_vlan):
            path = self.get_delete_vlan_config_attr(vlan_id, 'description')
        else:
            path = url.format(vlan_id)
        request = {'path': path, 'method': method}
        requests.append(request)
    return requests