from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def default_interface(self, ifname):
    """default_interface"""
    change = False
    xmlstr = ''
    self.updates_cmd.append('interface %s' % ifname)
    if self.intf_info['ifDescr']:
        xmlstr += CE_NC_XML_MERGE_INTF_DES % (ifname, '')
        self.updates_cmd.append('undo description')
        change = True
    if is_admin_state_enable(self.intf_type) and self.intf_info['ifAdminStatus'] != 'up':
        xmlstr += CE_NC_XML_MERGE_INTF_STATUS % (ifname, 'up')
        self.updates_cmd.append('undo shutdown')
        change = True
    if is_portswitch_enalbe(self.intf_type) and self.intf_info['isL2SwitchPort'] != 'true':
        xmlstr += CE_NC_XML_MERGE_INTF_L2ENABLE % (ifname, 'enable')
        self.updates_cmd.append('portswitch')
        change = True
    if not change:
        return
    conf_str = '<config> ' + xmlstr + ' </config>'
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'SET_INTF_DEFAULT')
    self.changed = True