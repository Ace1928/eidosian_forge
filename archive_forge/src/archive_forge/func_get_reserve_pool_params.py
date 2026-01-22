from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_reserve_pool_params(self, pool_info):
    """
        Process Reserved Pool parameters from playbook data
        for Reserved Pool configuration in Cisco Catalyst Center

        Parameters:
            pool_info (dict) - Playbook data containing information about the reserved pool

        Returns:
            reserve_pool (dict) - Processed Reserved pool data
            in the format suitable for the Cisco Catalyst Center config
        """
    reserve_pool = {'name': pool_info.get('groupName'), 'site_id': pool_info.get('siteId')}
    if len(pool_info.get('ipPools')) == 1:
        reserve_pool.update({'ipv4DhcpServers': pool_info.get('ipPools')[0].get('dhcpServerIps'), 'ipv4DnsServers': pool_info.get('ipPools')[0].get('dnsServerIps'), 'ipv6AddressSpace': 'False'})
        if pool_info.get('ipPools')[0].get('gateways') != []:
            reserve_pool.update({'ipv4GateWay': pool_info.get('ipPools')[0].get('gateways')[0]})
        else:
            reserve_pool.update({'ipv4GateWay': ''})
        reserve_pool.update({'ipv6AddressSpace': 'False'})
    elif len(pool_info.get('ipPools')) == 2:
        if not pool_info.get('ipPools')[0].get('ipv6'):
            reserve_pool.update({'ipv4DhcpServers': pool_info.get('ipPools')[0].get('dhcpServerIps'), 'ipv4DnsServers': pool_info.get('ipPools')[0].get('dnsServerIps'), 'ipv6AddressSpace': 'True', 'ipv6DhcpServers': pool_info.get('ipPools')[1].get('dhcpServerIps'), 'ipv6DnsServers': pool_info.get('ipPools')[1].get('dnsServerIps')})
            if pool_info.get('ipPools')[0].get('gateways') != []:
                reserve_pool.update({'ipv4GateWay': pool_info.get('ipPools')[0].get('gateways')[0]})
            else:
                reserve_pool.update({'ipv4GateWay': ''})
            if pool_info.get('ipPools')[1].get('gateways') != []:
                reserve_pool.update({'ipv6GateWay': pool_info.get('ipPools')[1].get('gateways')[0]})
            else:
                reserve_pool.update({'ipv6GateWay': ''})
        elif not pool_info.get('ipPools')[1].get('ipv6'):
            reserve_pool.update({'ipv4DhcpServers': pool_info.get('ipPools')[1].get('dhcpServerIps'), 'ipv4DnsServers': pool_info.get('ipPools')[1].get('dnsServerIps'), 'ipv6AddressSpace': 'True', 'ipv6DnsServers': pool_info.get('ipPools')[0].get('dnsServerIps'), 'ipv6DhcpServers': pool_info.get('ipPools')[0].get('dhcpServerIps')})
            if pool_info.get('ipPools')[1].get('gateways') != []:
                reserve_pool.update({'ipv4GateWay': pool_info.get('ipPools')[1].get('gateways')[0]})
            else:
                reserve_pool.update({'ipv4GateWay': ''})
            if pool_info.get('ipPools')[0].get('gateways') != []:
                reserve_pool.update({'ipv6GateWay': pool_info.get('ipPools')[0].get('gateways')[0]})
            else:
                reserve_pool.update({'ipv6GateWay': ''})
    reserve_pool.update({'slaacSupport': True})
    self.log('Formatted reserve pool details: {0}'.format(reserve_pool), 'DEBUG')
    return reserve_pool