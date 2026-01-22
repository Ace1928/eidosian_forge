from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_lldp_enable_pre_config(self):
    """Get lldp enable configure"""
    lldp_dict = dict()
    lldp_config = list()
    conf_enable_str = CE_NC_GET_GLOBAL_LLDPENABLE_CONFIG
    conf_enable_obj = get_nc_config(self.module, conf_enable_str)
    xml_enable_str = conf_enable_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root_enable = ElementTree.fromstring(xml_enable_str)
    ntpsite_enable = root_enable.findall('lldp/lldpSys')
    for nexthop_enable in ntpsite_enable:
        for ele_enable in nexthop_enable:
            if ele_enable.tag in ['lldpEnable']:
                lldp_dict[ele_enable.tag] = ele_enable.text
                if lldp_dict['lldpEnable'] == 'enabled':
                    self.enable_flag = 1
        lldp_config.append(dict(lldpenable=lldp_dict['lldpEnable']))
    return lldp_config