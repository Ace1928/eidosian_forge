from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_mac_dampening_attr(self, vrf_name, attr):
    url = '%s=%s/openconfig-mac-dampening:mac-dampening/config/%s' % (NETWORK_INSTANCE_PATH, vrf_name, attr)
    request = {'path': url, 'method': DELETE}
    return request