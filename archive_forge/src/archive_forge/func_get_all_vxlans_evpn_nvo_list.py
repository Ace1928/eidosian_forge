from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_vxlans_evpn_nvo_list(self):
    """Get all the evpn nvo list available """
    request = [{'path': 'data/sonic-vxlan:sonic-vxlan/EVPN_NVO/EVPN_NVO_LIST', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    vxlans_evpn_nvo_list = []
    if 'sonic-vxlan:EVPN_NVO_LIST' in response[0][1]:
        vxlans_evpn_nvo_list = response[0][1].get('sonic-vxlan:EVPN_NVO_LIST', [])
    return vxlans_evpn_nvo_list