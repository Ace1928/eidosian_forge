from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_options_subnet_range_ipv6(self):
    self.network_client.create_subnet.return_value = self._subnet_ipv6
    self._network.id = self._subnet_ipv6.network_id
    arglist = [self._subnet_ipv6.name, '--subnet-range', self._subnet_ipv6.cidr, '--network', self._subnet_ipv6.network_id, '--ip-version', str(self._subnet_ipv6.ip_version), '--ipv6-ra-mode', self._subnet_ipv6.ipv6_ra_mode, '--ipv6-address-mode', self._subnet_ipv6.ipv6_address_mode, '--gateway', self._subnet_ipv6.gateway_ip, '--dhcp']
    for dns_addr in self._subnet_ipv6.dns_nameservers:
        arglist.append('--dns-nameserver')
        arglist.append(dns_addr)
    for host_route in self._subnet_ipv6.host_routes:
        arglist.append('--host-route')
        value = 'gateway=' + host_route.get('nexthop', '') + ',destination=' + host_route.get('destination', '')
        arglist.append(value)
    for pool in self._subnet_ipv6.allocation_pools:
        arglist.append('--allocation-pool')
        value = 'start=' + pool.get('start', '') + ',end=' + pool.get('end', '')
        arglist.append(value)
    for service_type in self._subnet_ipv6.service_types:
        arglist.append('--service-type')
        arglist.append(service_type)
    verifylist = [('name', self._subnet_ipv6.name), ('subnet_range', self._subnet_ipv6.cidr), ('network', self._subnet_ipv6.network_id), ('ip_version', self._subnet_ipv6.ip_version), ('ipv6_ra_mode', self._subnet_ipv6.ipv6_ra_mode), ('ipv6_address_mode', self._subnet_ipv6.ipv6_address_mode), ('gateway', self._subnet_ipv6.gateway_ip), ('dns_nameservers', self._subnet_ipv6.dns_nameservers), ('dhcp', self._subnet_ipv6.enable_dhcp), ('host_routes', subnet_v2.convert_entries_to_gateway(self._subnet_ipv6.host_routes)), ('allocation_pools', self._subnet_ipv6.allocation_pools), ('service_types', self._subnet_ipv6.service_types)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet.assert_called_once_with(**{'cidr': self._subnet_ipv6.cidr, 'dns_nameservers': self._subnet_ipv6.dns_nameservers, 'enable_dhcp': self._subnet_ipv6.enable_dhcp, 'gateway_ip': self._subnet_ipv6.gateway_ip, 'host_routes': self._subnet_ipv6.host_routes, 'ip_version': self._subnet_ipv6.ip_version, 'ipv6_address_mode': self._subnet_ipv6.ipv6_address_mode, 'ipv6_ra_mode': self._subnet_ipv6.ipv6_ra_mode, 'name': self._subnet_ipv6.name, 'network_id': self._subnet_ipv6.network_id, 'allocation_pools': self._subnet_ipv6.allocation_pools, 'service_types': self._subnet_ipv6.service_types})
    self.assertFalse(self.network_client.set_tags.called)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data_ipv6, data)