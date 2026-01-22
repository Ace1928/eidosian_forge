from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
def get_access_delete_switchport_request(self, intf_name):
    """Returns the request as a dict to delete the access vlan
        configuration for the given interface
        """
    method = DELETE
    key = intf_key
    if intf_name.startswith('PortChannel'):
        key = port_chnl_key
    url = 'data/openconfig-interfaces:interfaces/interface={}/{}/openconfig-vlan:switched-vlan/config/access-vlan'
    request = {'path': url.format(intf_name, key), 'method': method}
    return request