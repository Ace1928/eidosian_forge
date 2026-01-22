from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def remove_nexthop(self):
    """remove ospf nexthop weight"""
    if not self.is_nexthop_exist():
        return
    xml_nh = CE_NC_XML_DELETE_NEXTHOP % self.nexthop_addr
    xml_topo = CE_NC_XML_BUILD_TOPO % xml_nh
    xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, xml_topo)
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'DELETE_NEXTHOP_WEIGHT')
    self.updates_cmd.append('ospf %s' % self.process_id)
    self.updates_cmd.append('undo nexthop %s' % self.nexthop_addr)
    self.changed = True