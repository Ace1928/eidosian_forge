from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def operate_static_route(self, version, prefix, mask, nhp_interface, next_hop, vrf, destvrf, state):
    """operate ipv4 static route"""
    description_xml = '\n'
    preference_xml = '\n'
    tag_xml = '\n'
    bfd_xml = '\n'
    if next_hop is None:
        next_hop = '0.0.0.0'
    if nhp_interface is None:
        nhp_interface = 'Invalid0'
    if vrf is None:
        vpn_instance = '_public_'
    else:
        vpn_instance = vrf
    if destvrf is None:
        dest_vpn_instance = '_public_'
    else:
        dest_vpn_instance = destvrf
    description_xml = get_xml(CE_NC_SET_DESCRIPTION, self.description)
    preference_xml = get_xml(CE_NC_SET_PREFERENCE, self.pref)
    tag_xml = get_xml(CE_NC_SET_TAG, self.tag)
    if self.function_flag == 'staticBFD':
        if self.bfd_session_name:
            bfd_xml = CE_NC_SET_BFDSESSIONNAME % self.bfd_session_name
    else:
        bfd_xml = CE_NC_SET_BFDENABLE
    if state == 'present':
        configxmlstr = CE_NC_SET_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop, description_xml, preference_xml, tag_xml, bfd_xml)
    else:
        configxmlstr = CE_NC_DELETE_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop)
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE')