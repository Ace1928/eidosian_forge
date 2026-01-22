from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config, execute_nc_action
def get_interface_mdn_exist_config(self):
    """Get lldp existed configure"""
    lldp_config = list()
    lldp_dict = dict()
    conf_enable_str = CE_NC_GET_GLOBAL_LLDPENABLE_CONFIG
    conf_enable_obj = get_nc_config(self.module, conf_enable_str)
    xml_enable_str = conf_enable_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root_enable = ElementTree.fromstring(xml_enable_str)
    ntpsite_enable = root_enable.findall('lldp/lldpSys')
    for nexthop_enable in ntpsite_enable:
        for ele_enable in nexthop_enable:
            if ele_enable.tag in ['lldpEnable']:
                lldp_dict[ele_enable.tag] = ele_enable.text
        if self.state == 'present':
            if lldp_dict['lldpEnable'] == 'enabled':
                self.enable_flag = 1
        lldp_config.append(dict(lldpenable=lldp_dict['lldpEnable']))
    if self.enable_flag == 1:
        conf_str = CE_NC_GET_INTERFACE_MDNENABLE_CONFIG
        conf_obj = get_nc_config(self.module, conf_str)
        if '<data/>' in conf_obj:
            return lldp_config
        xml_str = conf_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        ntpsite = root.findall('lldp/mdnInterfaces/mdnInterface')
        for nexthop in ntpsite:
            for ele in nexthop:
                if ele.tag in ['ifName', 'mdnStatus']:
                    lldp_dict[ele.tag] = ele.text
            if self.state == 'present':
                cur_interface_mdn_cfg = dict(ifname=lldp_dict['ifName'], mdnstatus=lldp_dict['mdnStatus'])
                exp_interface_mdn_cfg = dict(ifname=self.ifname, mdnstatus=self.mdnstatus)
                if self.ifname == lldp_dict['ifName']:
                    if cur_interface_mdn_cfg != exp_interface_mdn_cfg:
                        self.conf_exsit = True
                        lldp_config.append(dict(ifname=lldp_dict['ifName'], mdnstatus=lldp_dict['mdnStatus']))
                        return lldp_config
                    lldp_config.append(dict(ifname=lldp_dict['ifName'], mdnstatus=lldp_dict['mdnStatus']))
    return lldp_config