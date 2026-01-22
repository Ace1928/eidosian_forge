from __future__ import (absolute_import, division, print_function)
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def get_dldp_exist_config(self):
    """Get current dldp existed configuration"""
    dldp_conf = dict()
    xml_str = CE_NC_GET_GLOBAL_DLDP_CONFIG
    con_obj = get_nc_config(self.module, xml_str)
    if '<data/>' in con_obj:
        return dldp_conf
    xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    topo = root.find('dldp/dldpSys')
    if not topo:
        self.module.fail_json(msg='Error: Get current DLDP configuration failed.')
    for eles in topo:
        if eles.tag in ['dldpEnable', 'dldpInterval', 'dldpWorkMode', 'dldpAuthMode']:
            if eles.tag == 'dldpEnable':
                if eles.text == 'true':
                    value = 'enable'
                else:
                    value = 'disable'
            else:
                value = eles.text
            dldp_conf[eles.tag] = value
    return dldp_conf