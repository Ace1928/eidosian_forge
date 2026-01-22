from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_global_bfd(self, state):
    """get ipv4 global bfd"""
    self.static_routes_info['sroute_global_bfd'] = list()
    getglobalbfdxmlstr = None
    if self.aftype == 'v4':
        getglobalbfdxmlstr = CE_NC_GET_STATIC_ROUTE_IPV4_GLOBAL_BFD
    if getglobalbfdxmlstr is not None:
        xml_global_bfd_str = get_nc_config(self.module, getglobalbfdxmlstr)
        if 'data/' in xml_global_bfd_str:
            return
        xml_global_bfd_str = xml_global_bfd_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_global_bfd_str)
        static_routes_global_bfd = root.findall('staticrt/staticrtbase/srIPv4StaticSite')
        if static_routes_global_bfd:
            for static_route in static_routes_global_bfd:
                static_info = dict()
                for static_ele in static_route:
                    if static_ele.tag == 'minTxInterval':
                        if static_ele.text is not None:
                            static_info['minTxInterval'] = static_ele.text
                    if static_ele.tag == 'minRxInterval':
                        if static_ele.text is not None:
                            static_info['minRxInterval'] = static_ele.text
                    if static_ele.tag == 'multiplier':
                        if static_ele.text is not None:
                            static_info['multiplier'] = static_ele.text
                self.static_routes_info['sroute_global_bfd'].append(static_info)