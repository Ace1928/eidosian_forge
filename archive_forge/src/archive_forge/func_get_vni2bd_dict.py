from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_vni2bd_dict(self):
    """ get vni2bd attributes dict."""
    vni2bd_info = dict()
    conf_str = CE_NC_GET_VNI_BD_INFO
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return vni2bd_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    vni2bd_info['vni2BdInfos'] = list()
    vni2bds = root.findall('nvo3/nvo3Vni2Bds/nvo3Vni2Bd')
    if vni2bds:
        for vni2bd in vni2bds:
            vni_dict = dict()
            for ele in vni2bd:
                if ele.tag in ['vniId', 'bdId']:
                    vni_dict[ele.tag] = ele.text
            vni2bd_info['vni2BdInfos'].append(vni_dict)
    return vni2bd_info