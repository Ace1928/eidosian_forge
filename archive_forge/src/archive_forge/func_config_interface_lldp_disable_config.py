from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_interface_lldp_disable_config(self):
    if self.function_lldp_interface_flag == 'disableINTERFACE':
        if self.enable_flag == 1 and self.conf_interface_lldp_disable_exsit:
            if self.ifname:
                xml_str = CE_NC_MERGE_INTERFACE_LLDP_CONFIG % (self.ifname, self.lldpadminstatus)
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'INTERFACE_LLDP_DISABLE_CONFIG')
                self.changed = True