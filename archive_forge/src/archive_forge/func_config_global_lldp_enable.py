from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_global_lldp_enable(self):
    if self.state == 'present':
        if self.enable_flag == 0 and self.lldpenable == 'enabled':
            xml_str = CE_NC_MERGE_GLOBA_LLDPENABLE_CONFIG % self.lldpenable
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'LLDP_ENABLE_CONFIG')
            self.changed = True
        elif self.enable_flag == 1 and self.lldpenable == 'disabled':
            xml_str = CE_NC_MERGE_GLOBA_LLDPENABLE_CONFIG % self.lldpenable
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'LLDP_ENABLE_CONFIG')
            self.changed = True