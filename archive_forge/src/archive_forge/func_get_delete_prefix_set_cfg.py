from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_prefix_set_cfg(self, command):
    """Create and return a REST API request to delete the prefix set specified
        by the current "command"."""
    pfx_set_name = command.get('name', None)
    requests = [{'path': self.prefix_set_delete_uri.format(pfx_set_name), 'method': DELETE}]
    return requests