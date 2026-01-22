from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_port_info(self, interface):
    """Get port information"""
    if_type = get_interface_type(interface)
    if if_type == 'meth':
        xml_str = CE_NC_GET_PORT_SPEED % interface.lower().replace('meth', 'MEth')
    else:
        xml_str = CE_NC_GET_PORT_SPEED % interface.upper()
    con_obj = get_nc_config(self.module, xml_str)
    if '<data/>' in con_obj:
        return
    xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    port_info = root.find('devm/ports/port')
    if port_info:
        for eles in port_info:
            if eles.tag == 'ethernetPort':
                for ele in eles:
                    if ele.tag == 'speed':
                        self.result[interface]['Speed'] = ele.text