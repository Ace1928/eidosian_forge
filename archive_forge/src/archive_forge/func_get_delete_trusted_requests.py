from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_trusted_requests(self, afi):
    """makes and returns request to delete the given trusted interfaces for the given afi faimily.
        input expected as a dictionary of form {"afi": <ip_version>, "trusted": [{"intf_name": <name>}...]}"""
    requests = []
    if afi.get('trusted'):
        for intf in afi.get('trusted'):
            intf_name = intf.get('intf_name')
            if intf_name:
                requests.append({'path': self.trusted_uri.format(name=intf_name, v=self.afi_to_vnum(afi)), 'method': self.delete_method_value})
    return requests