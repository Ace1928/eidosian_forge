from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.mclag.mclag import MclagArgs
from ansible.module_utils.connection import ConnectionError
def get_all_mclag(self):
    """Get all the mclag available in chassis"""
    request = [{'path': 'data/openconfig-mclag:mclag', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'openconfig-mclag:mclag' in response[0][1]:
        data = response[0][1]['openconfig-mclag:mclag']
    else:
        data = {}
    return data