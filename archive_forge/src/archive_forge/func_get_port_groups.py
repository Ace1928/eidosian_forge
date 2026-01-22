from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.port_group.port_group import Port_groupArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_port_groups(self):
    """Get all the port group configurations"""
    pgs_request = [{'path': 'data/openconfig-port-group:port-groups/port-group', 'method': GET}]
    try:
        pgs_response = edit_config(self._module, to_request(self._module, pgs_request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    pgs_config = []
    if 'openconfig-port-group:port-group' in pgs_response[0][1]:
        pgs_config = pgs_response[0][1].get('openconfig-port-group:port-group', [])
    pgs = []
    for pg_config in pgs_config:
        pg = dict()
        if 'config' in pg_config:
            pg['id'] = pg_config['id']
            speed_str = pg_config['config'].get('speed', None)
            if speed_str:
                pg['speed'] = speed_str.split(':', 1)[-1]
                pgs.append(pg)
    return pgs