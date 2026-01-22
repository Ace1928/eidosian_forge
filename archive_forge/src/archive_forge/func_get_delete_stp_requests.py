from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_stp_requests(self, commands, have, is_delete_all):
    requests = []
    if not commands:
        return requests
    if is_delete_all:
        requests.append(self.get_delete_all_stp_request())
    else:
        requests.extend(self.get_delete_stp_mstp_requests(commands, have))
        requests.extend(self.get_delete_stp_pvst_requests(commands, have))
        requests.extend(self.get_delete_stp_rapid_pvst_requests(commands, have))
        requests.extend(self.get_delete_stp_interfaces_requests(commands, have))
        requests.extend(self.get_delete_stp_global_requests(commands, have))
    return requests