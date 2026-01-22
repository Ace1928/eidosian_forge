from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.dhcp_snooping.dhcp_snooping import Dhcp_snoopingArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_dhcp_snooping(self):
    config = {}
    config['top_level'] = self.get_dhcp_snooping_top_level()
    config['binding'] = self.get_dhcp_snooping_binding()
    return config