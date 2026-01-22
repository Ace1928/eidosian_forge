from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_interfaces(self, iftype):
    """ Delete interfaces with type."""
    xmlstr = ''
    intfs_list = self.intfs_info.get(iftype.lower())
    if not intfs_list:
        return
    for intf in intfs_list:
        xmlstr += CE_NC_XML_DELETE_INTF % intf['ifName']
        self.updates_cmd.append('undo interface %s' % intf['ifName'])
    conf_str = '<config> ' + xmlstr + ' </config>'
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'DELETE_INTFS')
    self.changed = True