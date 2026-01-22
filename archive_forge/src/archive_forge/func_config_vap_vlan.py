from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_vap_vlan(self):
    """configure a VLAN as a service access point"""
    xml_str = ''
    if self.state == 'present':
        if not is_vlan_in_bitmap(self.bind_vlan_id, self.vap_info['vlanList']):
            self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
            self.updates_cmd.append('l2 binding vlan %s' % self.bind_vlan_id)
            vlan_bitmap = vlan_vid_to_bitmap(self.bind_vlan_id)
            xml_str = CE_NC_MERGE_BD_VLAN % (self.bridge_domain_id, vlan_bitmap, vlan_bitmap)
    elif is_vlan_in_bitmap(self.bind_vlan_id, self.vap_info['vlanList']):
        self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
        self.updates_cmd.append('undo l2 binding vlan %s' % self.bind_vlan_id)
        vlan_bitmap = vlan_vid_to_bitmap(self.bind_vlan_id)
        xml_str = CE_NC_MERGE_BD_VLAN % (self.bridge_domain_id, '0' * 1024, vlan_bitmap)
    if not xml_str:
        return
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'CONFIG_VAP_VLAN')
    self.changed = True