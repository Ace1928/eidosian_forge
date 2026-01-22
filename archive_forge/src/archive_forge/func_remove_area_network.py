from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def remove_area_network(self):
    """remvoe ospf area network"""
    if not self.is_network_exist():
        return
    xml_network = CE_NC_XML_DELETE_NETWORKS % (self.addr, self.get_wildcard_mask())
    xml_area = CE_NC_XML_BUILD_AREA % (self.get_area_ip(), xml_network)
    xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, xml_area)
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'DELETE_AREA_NETWORK')
    self.updates_cmd.append('ospf %s' % self.process_id)
    self.updates_cmd.append('area %s' % self.get_area_ip())
    self.updates_cmd.append('undo network %s %s' % (self.addr, self.get_wildcard_mask()))
    self.changed = True