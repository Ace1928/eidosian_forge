from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_l2_sub_intf_dict(self, ifname):
    """get l2 sub-interface info"""
    intf_info = dict()
    if not ifname:
        return intf_info
    conf_str = CE_NC_GET_ENCAP % ifname
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return intf_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    bds = root.find('ethernet/servicePoints/servicePoint')
    if not bds:
        return intf_info
    for ele in bds:
        if ele.tag in ['ifName', 'flowType']:
            intf_info[ele.tag] = ele.text.lower()
    if intf_info.get('flowType') == 'dot1q':
        ce_vid = root.find('ethernet/servicePoints/servicePoint/flowDot1qs')
        intf_info['dot1qVids'] = ''
        if ce_vid:
            for ele in ce_vid:
                if ele.tag == 'dot1qVids':
                    intf_info['dot1qVids'] = ele.text
    elif intf_info.get('flowType') == 'qinq':
        vids = root.find('ethernet/servicePoints/servicePoint/flowQinqs/flowQinq')
        if vids:
            for ele in vids:
                if ele.tag in ['peVlanId', 'ceVids']:
                    intf_info[ele.tag] = ele.text
    return intf_info