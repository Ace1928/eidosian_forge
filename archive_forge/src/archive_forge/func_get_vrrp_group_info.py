from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_vrrp_group_info(self):
    """ get vrrp group info."""
    vrrp_group_info = dict()
    conf_str = CE_NC_GET_VRRP_GROUP_INFO % (self.interface, self.vrid)
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return vrrp_group_info
    else:
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        global_info = root.findall('vrrp/vrrpGroups/vrrpGroup')
        if global_info:
            for tmp in global_info:
                for site in tmp:
                    if site.tag in ['ifName', 'vrrpId', 'priority', 'advertiseInterval', 'preemptMode', 'delayTime', 'authenticationMode', 'authenticationKey', 'vrrpType', 'adminVrrpId', 'adminIfName', 'adminIgnoreIfDown', 'isPlain', 'unflowdown', 'fastResume', 'holdMultiplier']:
                        vrrp_group_info[site.tag] = site.text
        return vrrp_group_info