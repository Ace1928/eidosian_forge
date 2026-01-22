from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_vrf_af(self):
    """ check if vrf is need to change"""
    self.vrf_af_info['vpnInstAF'] = list()
    if self.evpn is True:
        getxmlstr = CE_NC_GET_VRF_AF % (self.vrf, CE_NC_GET_EXTEND_VRF_TARGET)
    else:
        getxmlstr = CE_NC_GET_VRF_AF % (self.vrf, CE_NC_GET_VRF_TARGET)
    xml_str = get_nc_config(self.module, getxmlstr)
    if 'data/' in xml_str:
        return self.state == 'present'
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    vrf_addr_types = root.findall('l3vpn/l3vpncomm/l3vpnInstances/l3vpnInstance/vpnInstAFs/vpnInstAF')
    if vrf_addr_types:
        for vrf_addr_type in vrf_addr_types:
            vrf_af_info = dict()
            for vrf_addr_type_ele in vrf_addr_type:
                if vrf_addr_type_ele.tag in ['vrfName', 'afType', 'vrfRD']:
                    vrf_af_info[vrf_addr_type_ele.tag] = vrf_addr_type_ele.text
                if vrf_addr_type_ele.tag == 'vpnTargets':
                    vrf_af_info['vpnTargets'] = list()
                    for rtargets in vrf_addr_type_ele:
                        rt_dict = dict()
                        for rtarget in rtargets:
                            if rtarget.tag in ['vrfRTValue', 'vrfRTType']:
                                rt_dict[rtarget.tag] = rtarget.text
                        vrf_af_info['vpnTargets'].append(rt_dict)
                if vrf_addr_type_ele.tag == 'exVpnTargets':
                    vrf_af_info['evpnTargets'] = list()
                    for rtargets in vrf_addr_type_ele:
                        rt_dict = dict()
                        for rtarget in rtargets:
                            if rtarget.tag in ['vrfRTValue', 'vrfRTType']:
                                rt_dict[rtarget.tag] = rtarget.text
                        vrf_af_info['evpnTargets'].append(rt_dict)
            self.vrf_af_info['vpnInstAF'].append(vrf_af_info)