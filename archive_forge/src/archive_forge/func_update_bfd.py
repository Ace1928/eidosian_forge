from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bfd.bfd import BfdArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_bfd(self, data):
    bfd_dict = {}
    if data:
        bfd_dict['profiles'] = self.update_profiles(data)
        bfd_dict['single_hops'] = self.update_single_hops(data)
        bfd_dict['multi_hops'] = self.update_multi_hops(data)
    return bfd_dict