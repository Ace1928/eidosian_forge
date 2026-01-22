from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def merge_vlan(self, vlan_id, name, description):
    """Merge vlan."""
    conf_str = None
    if not name and description:
        conf_str = CE_NC_MERGE_VLAN_DES % (vlan_id, description)
    if not description and name:
        conf_str = CE_NC_MERGE_VLAN_NAME % (vlan_id, name)
    if description and name:
        conf_str = CE_NC_MERGE_VLAN % (vlan_id, name, description)
    if not conf_str:
        return
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'MERGE_VLAN')
    self.changed = True