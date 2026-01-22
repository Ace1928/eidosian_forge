from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_process(self):
    """Delete ospf process"""
    xml_str = CE_NC_DELETE_PROCESS % self.process_id
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'DELETE_PROCESS')
    self.updates_cmd.append('undo ospf %s' % self.process_id)
    self.changed = True