from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_vxlans_tunnels_vlan_map(self):
    """Get all the vxlan tunnels and vlan map available """
    request = [{'path': 'data/sonic-vxlan:sonic-vxlan', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    vxlans_tunnels_vlan_map = {}
    if 'sonic-vxlan:sonic-vxlan' in response[0][1]:
        vxlans_tunnels_vlan_map = response[0][1].get('sonic-vxlan:sonic-vxlan', {})
    return vxlans_tunnels_vlan_map