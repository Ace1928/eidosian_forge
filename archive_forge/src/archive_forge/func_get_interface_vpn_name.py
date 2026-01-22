from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def get_interface_vpn_name(self, vpninfo, vpn_name):
    """ get vpn instance name"""
    l3vpn_if = vpninfo.findall('l3vpnIf')
    for l3vpn_ifinfo in l3vpn_if:
        for ele in l3vpn_ifinfo:
            if ele.tag in ['ifName']:
                if ele.text.lower() == self.vpn_interface.lower():
                    self.intf_info['vrfName'] = vpn_name