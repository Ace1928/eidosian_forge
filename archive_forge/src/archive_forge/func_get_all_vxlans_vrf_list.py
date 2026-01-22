from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_vxlans_vrf_list(self):
    """Get all the vxlan tunnels and vlan map available """
    request = [{'path': 'data/sonic-vrf:sonic-vrf/VRF/VRF_LIST', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-vrf:VRF_LIST' in response[0][1]:
        vxlan_vrf_list = response[0][1].get('sonic-vrf:VRF_LIST', {})
    return vxlan_vrf_list