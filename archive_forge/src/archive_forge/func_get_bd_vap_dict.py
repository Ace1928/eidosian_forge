from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_bd_vap_dict(self):
    """get virtual access point info"""
    vap_info = dict()
    conf_str = CE_NC_GET_BD_VAP % self.bridge_domain_id
    xml_str = get_nc_config(self.module, conf_str)
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    vap_info['bdId'] = self.bridge_domain_id
    root = ElementTree.fromstring(xml_str)
    vap_info['vlanList'] = ''
    vap_vlan = root.find('evc/bds/bd/bdBindVlan')
    if vap_vlan:
        for ele in vap_vlan:
            if ele.tag == 'vlanList':
                vap_info['vlanList'] = ele.text
    vap_ifs = root.findall('evc/bds/bd/servicePoints/servicePoint/ifName')
    if_list = list()
    if vap_ifs:
        for vap_if in vap_ifs:
            if vap_if.tag == 'ifName':
                if_list.append(vap_if.text)
    vap_info['intfList'] = if_list
    return vap_info