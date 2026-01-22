from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_interface_lldp_disable_pre_config(self):
    """Get interface undo lldp disable configure"""
    lldp_dict = dict()
    interface_lldp_disable_dict = dict()
    if self.enable_flag == 1:
        conf_enable_str = CE_NC_GET_INTERFACE_LLDP_CONFIG
        conf_enable_obj = get_nc_config(self.module, conf_enable_str)
        if '<data/>' in conf_enable_obj:
            return
        xml_enable_str = conf_enable_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_enable_str)
        lldp_disable_enable = root.findall('lldp/lldpInterfaces/lldpInterface')
        for nexthop_enable in lldp_disable_enable:
            name = nexthop_enable.find('ifName')
            status = nexthop_enable.find('lldpAdminStatus')
            if name is not None and status is not None:
                interface_lldp_disable_dict[name.text] = status.text
    return interface_lldp_disable_dict