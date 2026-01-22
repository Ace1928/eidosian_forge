from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def get_dldp_intf_exist_config(self):
    """Get current dldp existed config"""
    dldp_conf = dict()
    xml_str = CE_NC_GET_INTF_DLDP_CONFIG % self.interface
    con_obj = get_nc_config(self.module, xml_str)
    if '<data/>' in con_obj:
        dldp_conf['dldpEnable'] = 'disable'
        dldp_conf['dldpCompatibleEnable'] = ''
        dldp_conf['dldpLocalMac'] = ''
        return dldp_conf
    xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    topo = root.find('dldp/dldpInterfaces/dldpInterface')
    if topo is None:
        self.module.fail_json(msg='Error: Get current DLDP configuration failed.')
    for eles in topo:
        if eles.tag in ['dldpEnable', 'dldpCompatibleEnable', 'dldpLocalMac']:
            if not eles.text:
                dldp_conf[eles.tag] = ''
            else:
                if eles.tag == 'dldpEnable' or eles.tag == 'dldpCompatibleEnable':
                    if eles.text == 'true':
                        value = 'enable'
                    else:
                        value = 'disable'
                else:
                    value = eles.text
                dldp_conf[eles.tag] = value
    return dldp_conf