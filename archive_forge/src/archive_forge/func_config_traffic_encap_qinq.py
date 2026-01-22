from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_traffic_encap_qinq(self):
    """configure traffic encapsulation type qinq"""
    xml_str = ''
    self.updates_cmd.append('interface %s' % self.l2_sub_interface)
    if self.state == 'present':
        if self.encapsulation != self.l2sub_info.get('flowType'):
            if self.ce_vid:
                vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                xml_str = CE_NC_SET_ENCAP_QINQ % (self.l2_sub_interface, self.pe_vid, vlan_bitmap, vlan_bitmap)
                self.updates_cmd.append('encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
            else:
                xml_str = CE_NC_SET_ENCAP % (self.l2_sub_interface, self.encapsulation)
                self.updates_cmd.append('encapsulation %s' % self.encapsulation)
        elif self.ce_vid:
            if not is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('ceVids')) or self.pe_vid != self.l2sub_info.get('peVlanId'):
                vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                xml_str = CE_NC_SET_ENCAP_QINQ % (self.l2_sub_interface, self.pe_vid, vlan_bitmap, vlan_bitmap)
                self.updates_cmd.append('encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
    elif self.encapsulation == self.l2sub_info.get('flowType'):
        if self.ce_vid:
            if is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('ceVids')) and self.pe_vid == self.l2sub_info.get('peVlanId'):
                xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                self.updates_cmd.append('undo encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
        else:
            xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
            self.updates_cmd.append('undo encapsulation %s' % self.encapsulation)
    if not xml_str:
        self.updates_cmd.pop()
        return
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'CONFIG_INTF_ENCAP_QINQ')
    self.changed = True