from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_modify_radius_server_requests(self, command, have):
    requests = []
    if not command:
        return requests
    request = self.get_modify_global_config_request(command)
    if request:
        requests.append(request)
    request = self.get_modify_global_ext_config_request(command)
    if request:
        requests.append(request)
    request = self.get_modify_servers_request(command)
    if request:
        requests.append(request)
    return requests