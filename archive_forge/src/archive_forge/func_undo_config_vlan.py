from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def undo_config_vlan(self, vlanid):
    """Delete vlan."""
    conf_str = CE_NC_DELETE_VLAN % vlanid
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'DELETE_VLAN')
    self.changed = True
    self.updates_cmd.append('undo vlan %s' % self.vlan_id)