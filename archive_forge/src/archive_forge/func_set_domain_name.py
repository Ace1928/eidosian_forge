from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def set_domain_name(self, attr):
    key = attr['mgr_attr_name']
    nic_info = self.get_manager_ethernet_uri()
    ethuri = nic_info['nic_addr']
    response = self.get_request(self.root_uri + ethuri)
    if not response['ret']:
        return response
    data = response['data']
    payload = {'DHCPv4': {'UseDomainName': ''}}
    if data['DHCPv4']['UseDomainName']:
        payload['DHCPv4']['UseDomainName'] = False
        res_dhv4 = self.patch_request(self.root_uri + ethuri, payload)
        if not res_dhv4['ret']:
            return res_dhv4
    payload = {'DHCPv6': {'UseDomainName': ''}}
    if data['DHCPv6']['UseDomainName']:
        payload['DHCPv6']['UseDomainName'] = False
        res_dhv6 = self.patch_request(self.root_uri + ethuri, payload)
        if not res_dhv6['ret']:
            return res_dhv6
    domain_name = attr['mgr_attr_value']
    payload = {'Oem': {'Hpe': {key: domain_name}}}
    response = self.patch_request(self.root_uri + ethuri, payload)
    if not response['ret']:
        return response
    return {'ret': True, 'changed': True, 'msg': 'Modified %s' % attr['mgr_attr_name']}