from __future__ import absolute_import, division, print_function
from natsort import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_port_group_default_speed(self):
    """Get all the port group default speeds"""
    pgs_request = [{'path': 'data/openconfig-port-group:port-groups/port-group', 'method': GET}]
    try:
        pgs_response = edit_config(self._module, to_request(self._module, pgs_request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    pgs_config = []
    if 'openconfig-port-group:port-group' in pgs_response[0][1]:
        pgs_config = pgs_response[0][1].get('openconfig-port-group:port-group', [])
    pgs_dft_speeds = []
    for pg_config in pgs_config:
        pg_state = dict()
        if 'state' in pg_config:
            pg_state['id'] = pg_config['id']
            dft_speed_str = pg_config['state'].get('default-speed', None)
            if dft_speed_str:
                pg_state['speed'] = dft_speed_str.split(':', 1)[-1]
                pgs_dft_speeds.append(pg_state)
    return pgs_dft_speeds