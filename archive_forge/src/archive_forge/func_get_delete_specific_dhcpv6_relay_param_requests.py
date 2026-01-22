from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_specific_dhcpv6_relay_param_requests(self, command, have, is_state_deleted=True):
    """Get requests to delete specific DHCPv6 relay configurations
        based on the command specified for the interface
        """
    requests = []
    name = command['name']
    ipv6 = command.get('ipv6')
    have_ipv6 = have.get('ipv6')
    if not ipv6 or not have_ipv6:
        return requests
    server_addresses = self.get_server_addresses(ipv6.get('server_addresses'))
    have_server_addresses = self.get_server_addresses(have_ipv6.get('server_addresses'))
    if ipv6.get('server_addresses') and len(ipv6.get('server_addresses')) and (not server_addresses):
        requests.append(self.get_delete_all_dhcpv6_relay_intf_request(name))
        return requests
    del_server_addresses = have_server_addresses.intersection(server_addresses)
    if del_server_addresses:
        if is_state_deleted and len(del_server_addresses) == len(have_server_addresses):
            requests.append(self.get_delete_all_dhcpv6_relay_intf_request(name))
            return requests
        for addr in del_server_addresses:
            url = self.dhcpv6_relay_intf_config_path['server_address'].format(intf_name=name, server_address=addr)
            requests.append({'path': url, 'method': DELETE})
    if ipv6.get('source_interface') and have_ipv6.get('source_interface') and (ipv6['source_interface'] == have_ipv6['source_interface']):
        url = self.dhcpv6_relay_intf_config_path['source_interface'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv6.get('max_hop_count') and have_ipv6.get('max_hop_count') and (ipv6['max_hop_count'] == have_ipv6['max_hop_count']) and (have_ipv6['max_hop_count'] != DEFAULT_MAX_HOP_COUNT):
        url = self.dhcpv6_relay_intf_config_path['max_hop_count'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv6.get('vrf_select') is not None and have_ipv6.get('vrf_select'):
        url = self.dhcpv6_relay_intf_config_path['vrf_select'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    return requests