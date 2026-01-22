from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_eth_trunk(self):
    """Delete Eth-Trunk interface and remove all member"""
    if not self.trunk_info:
        return
    xml_str = ''
    mem_str = ''
    if self.trunk_info['TrunkMemberIfs']:
        for mem in self.trunk_info['TrunkMemberIfs']:
            mem_str += CE_NC_XML_DELETE_MEMBER % mem['memberIfName']
            self.updates_cmd.append('interface %s' % mem['memberIfName'])
            self.updates_cmd.append('undo eth-trunk')
        if mem_str:
            xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_str)
    xml_str += CE_NC_XML_DELETE_TRUNK % self.trunk_id
    self.updates_cmd.append('undo interface Eth-Trunk %s' % self.trunk_id)
    cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
    self.netconf_set_config(cfg_xml, 'DELETE_TRUNK')
    self.changed = True