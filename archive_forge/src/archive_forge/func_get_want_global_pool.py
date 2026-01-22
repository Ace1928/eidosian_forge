from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_want_global_pool(self, global_ippool):
    """
        Get all the Global Pool information from playbook
        Set the status and the msg before returning from the API
        Check the return value of the API with check_return_status()

        Parameters:
            global_ippool (dict) - Playbook global pool details containing IpAddressSpace,
            DHCP server IPs, DNS server IPs, IP pool name, IP pool CIDR, gateway, and type.

        Returns:
            self - The current object with updated desired Global Pool information.
        """
    want_global = {'settings': {'ippool': [{'IpAddressSpace': global_ippool.get('ip_address_space'), 'dhcpServerIps': global_ippool.get('dhcp_server_ips'), 'dnsServerIps': global_ippool.get('dns_server_ips'), 'ipPoolName': global_ippool.get('name'), 'ipPoolCidr': global_ippool.get('cidr'), 'gateway': global_ippool.get('gateway'), 'type': global_ippool.get('pool_type')}]}}
    want_ippool = want_global.get('settings').get('ippool')[0]
    if not self.have.get('globalPool').get('exists'):
        if want_ippool.get('dhcpServerIps') is None:
            want_ippool.update({'dhcpServerIps': []})
        if want_ippool.get('dnsServerIps') is None:
            want_ippool.update({'dnsServerIps': []})
        if want_ippool.get('IpAddressSpace') is None:
            want_ippool.update({'IpAddressSpace': ''})
        if want_ippool.get('gateway') is None:
            want_ippool.update({'gateway': ''})
        if want_ippool.get('type') is None:
            want_ippool.update({'type': 'Generic'})
    else:
        have_ippool = self.have.get('globalPool').get('details').get('settings').get('ippool')[0]
        want_ippool.update({'IpAddressSpace': have_ippool.get('IpAddressSpace'), 'type': have_ippool.get('type'), 'ipPoolCidr': have_ippool.get('ipPoolCidr')})
        want_ippool.update({})
        want_ippool.update({})
        for key in ['dhcpServerIps', 'dnsServerIps', 'gateway']:
            if want_ippool.get(key) is None and have_ippool.get(key) is not None:
                want_ippool[key] = have_ippool[key]
    self.log('Global pool playbook details: {0}'.format(want_global), 'DEBUG')
    self.want.update({'wantGlobal': want_global})
    self.msg = 'Collecting the global pool details from the playbook'
    self.status = 'success'
    return self