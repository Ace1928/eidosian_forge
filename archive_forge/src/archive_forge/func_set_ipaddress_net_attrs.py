from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def set_ipaddress_net_attrs(self, prefix_val, conf_afi):
    """Create and return a dictionary containing the values for any prefix-related
        attributes needed for handling of prefix configuration requests. NOTE: This
        method should be replaced with use of the Python "ipaddress" module after
        Ansible drops downward compatibility support for Python 2.7."""
    prefix_net = dict()
    if conf_afi == 'ipv4':
        prefix_net['max_prefixlen'] = 32
    else:
        prefix_net['max_prefixlen'] = 128
    prefix_net['prefixlen'] = int(prefix_val.split('/')[1])
    return prefix_net