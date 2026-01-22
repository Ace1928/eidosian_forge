from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def get_interface_interval_config(self):
    lldp_config = list()
    lldp_dict = dict()
    cur_interface_mdn_cfg = dict()
    exp_interface_mdn_cfg = dict()
    interface_lldp_disable_dict_tmp2 = self.get_interface_lldp_disable_pre_config()
    if self.enable_flag == 1:
        if interface_lldp_disable_dict_tmp2[self.ifname] != 'disabled':
            conf_str = CE_NC_GET_INTERFACE_INTERVAl_CONFIG
            conf_obj = get_nc_config(self.module, conf_str)
            if '<data/>' in conf_obj:
                return lldp_config
            xml_str = conf_obj.replace('\r', '').replace('\n', '')
            xml_str = xml_str.replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '')
            xml_str = xml_str.replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            txintervalsite = root.findall('lldp/lldpInterfaces/lldpInterface')
            for ele in txintervalsite:
                ifname_tmp = ele.find('ifName')
                txinterval_tmp = ele.find('msgInterval/txInterval')
                if ifname_tmp is not None:
                    if ifname_tmp.text is not None:
                        cur_interface_mdn_cfg['ifname'] = ifname_tmp.text
                if txinterval_tmp is not None:
                    if txinterval_tmp.text is not None:
                        cur_interface_mdn_cfg['txinterval'] = txinterval_tmp.text
                if self.state == 'present':
                    if self.ifname:
                        exp_interface_mdn_cfg['ifname'] = self.ifname
                        if self.txinterval:
                            exp_interface_mdn_cfg['txinterval'] = self.txinterval
                        if self.ifname == ifname_tmp.text:
                            key_list = exp_interface_mdn_cfg.keys()
                            key_list_cur = cur_interface_mdn_cfg.keys()
                            if len(key_list) != 0:
                                for key in key_list:
                                    if 'txinterval' == str(key) and self.ifname == cur_interface_mdn_cfg['ifname']:
                                        lldp_config.append(dict(ifname=cur_interface_mdn_cfg['ifname'], txinterval=exp_interface_mdn_cfg['txinterval']))
                                    if key in key_list_cur:
                                        if str(exp_interface_mdn_cfg[key]) != str(cur_interface_mdn_cfg[key]):
                                            self.conf_interval_exsit = True
                                            lldp_config.append(cur_interface_mdn_cfg)
                                            return lldp_config
                                    else:
                                        self.conf_interval_exsit = True
                                        return lldp_config
    return lldp_config