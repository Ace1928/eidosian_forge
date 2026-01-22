from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.stp.stp import StpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_stp_config(self, module):
    stp_cfg = None
    get_stp_path = '/data/openconfig-spanning-tree:stp'
    request = {'path': get_stp_path, 'method': 'get'}
    try:
        response = edit_config(module, to_request(module, request))
        stp_cfg = response[0][1].get('openconfig-spanning-tree:stp', None)
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    return stp_cfg