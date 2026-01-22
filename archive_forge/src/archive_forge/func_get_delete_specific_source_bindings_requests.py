from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_specific_source_bindings_requests(self, afi):
    """creates and builds a list of requests to delete the source bindings listed in the given afi family
        input expected as a dictionary of form to_delete = {"afi": <ip_version>, "source_bindings": <list of source_bindings>}"""
    requests = []
    for entry in afi.get('source_bindings'):
        if entry.get('mac_addr'):
            requests.append({'path': self.binding_uri + '={mac},{ipv}'.format(mac=entry.get('mac_addr'), ipv=afi.get('afi')), 'method': self.delete_method_value})
    return requests