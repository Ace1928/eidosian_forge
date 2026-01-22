from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def set_dns_server(self, attr):
    key = attr['mgr_attr_name']
    nic_info = self.get_manager_ethernet_uri()
    uri = nic_info['nic_addr']
    listofips = attr['mgr_attr_value'].split(' ')
    if len(listofips) > 3:
        return {'ret': False, 'changed': False, 'msg': 'More than 3 DNS Servers mentioned'}
    dns_list = []
    for ips in listofips:
        dns_list.append(ips)
    while len(dns_list) < 3:
        dns_list.append('0.0.0.0')
    payload = {'Oem': {'Hpe': {'IPv4': {key: dns_list}}}}
    response = self.patch_request(self.root_uri + uri, payload)
    if not response['ret']:
        return response
    return {'ret': True, 'changed': True, 'msg': 'Modified %s' % attr['mgr_attr_name']}