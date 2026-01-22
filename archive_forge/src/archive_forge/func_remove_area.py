from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def remove_area(self):
    """remove ospf area"""
    if not self.is_area_exist():
        return
    xml_area = CE_NC_XML_BUILD_DELETE_AREA % (self.get_area_ip(), '')
    xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, xml_area)
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'DELETE_AREA')
    self.updates_cmd.append('ospf %s' % self.process_id)
    self.updates_cmd.append('undo area %s' % self.get_area_ip())
    self.changed = True