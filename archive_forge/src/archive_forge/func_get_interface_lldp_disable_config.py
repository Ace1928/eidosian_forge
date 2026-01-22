from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_interface_lldp_disable_config(self):
    lldp_config = list()
    interface_lldp_disable_dict_tmp = dict()
    if self.state == 'present':
        if self.ifname:
            interface_lldp_disable_dict_tmp = self.get_interface_lldp_disable_pre_config()
            key_list = interface_lldp_disable_dict_tmp.keys()
            if len(key_list) != 0:
                for key in key_list:
                    if key == self.ifname:
                        if interface_lldp_disable_dict_tmp[key] != self.lldpadminstatus:
                            self.conf_interface_lldp_disable_exsit = True
                        else:
                            self.conf_interface_lldp_disable_exsit = False
            elif self.ifname not in key_list:
                self.conf_interface_lldp_disable_exsit = True
            elif len(key_list) == 0 and self.ifname and self.lldpadminstatus:
                self.conf_interface_lldp_disable_exsit = True
            lldp_config.append(interface_lldp_disable_dict_tmp)
    return lldp_config