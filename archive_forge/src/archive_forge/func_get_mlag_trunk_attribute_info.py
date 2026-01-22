from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_mlag_trunk_attribute_info(self):
    """ get mlag global info."""
    mlag_trunk_attribute_info = dict()
    eth_trunk = 'Eth-Trunk'
    eth_trunk += self.eth_trunk_id
    conf_str = CE_NC_GET_LACP_MLAG_INFO % eth_trunk
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return mlag_trunk_attribute_info
    else:
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        global_info = root.findall('./ifmtrunk/TrunkIfs/TrunkIf/lacpMlagIf')
        if global_info:
            for tmp in global_info:
                for site in tmp:
                    if site.tag in ['lacpMlagSysId', 'lacpMlagPriority']:
                        mlag_trunk_attribute_info[site.tag] = site.text
        return mlag_trunk_attribute_info