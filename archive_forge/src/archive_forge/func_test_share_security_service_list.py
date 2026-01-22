import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_list(self):
    arglist = ['--share-network', self.share_network.id, '--status', self.services_list[0].status, '--name', self.services_list[0].name, '--type', self.services_list[0].type, '--user', self.services_list[0].user, '--dns-ip', self.services_list[0].dns_ip, '--ou', self.services_list[0].ou, '--server', self.services_list[0].server, '--domain', self.services_list[0].domain, '--default-ad-site', self.services_list[0].default_ad_site, '--limit', '1']
    verifylist = [('share_network', self.share_network.id), ('status', self.services_list[0].status), ('name', self.services_list[0].name), ('type', self.services_list[0].type), ('user', self.services_list[0].user), ('dns_ip', self.services_list[0].dns_ip), ('ou', self.services_list[0].ou), ('server', self.services_list[0].server), ('domain', self.services_list[0].domain), ('default_ad_site', self.services_list[0].default_ad_site), ('limit', 1)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.security_services_mock.list.assert_called_with(search_opts={'all_tenants': False, 'status': self.services_list[0].status, 'name': self.services_list[0].name, 'type': self.services_list[0].type, 'user': self.services_list[0].user, 'dns_ip': self.services_list[0].dns_ip, 'server': self.services_list[0].server, 'domain': self.services_list[0].domain, 'default_ad_site': self.services_list[0].default_ad_site, 'offset': None, 'limit': 1, 'ou': self.services_list[0].ou, 'share_network_id': self.share_network.id}, detailed=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))