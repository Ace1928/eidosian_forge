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
def get_trunk_allowed_vlans_delete_switchport_request(self, intf_name, allowed_vlans):
    """Returns the request as a dict to delete the trunk vlan ranges
        specified in allowed_vlans for the given interface
        """
    method = DELETE
    vlan_id_list = ''
    for each_allowed_vlan in allowed_vlans:
        vlan_id = each_allowed_vlan['vlan']
        if '-' in vlan_id:
            vlan_id_fmt = vlan_id.replace('-', '..')
        else:
            vlan_id_fmt = vlan_id
        if vlan_id_list:
            vlan_id_list += ',{0}'.format(vlan_id_fmt)
        else:
            vlan_id_list = vlan_id_fmt
    key = intf_key
    if intf_name.startswith('PortChannel'):
        key = port_chnl_key
    url = 'data/openconfig-interfaces:interfaces/interface={0}/{1}/'.format(intf_name, key)
    url += 'openconfig-vlan:switched-vlan/config/'
    url += 'trunk-vlans=' + vlan_id_list.replace(',', '%2C')
    request = {'path': url, 'method': method}
    return request