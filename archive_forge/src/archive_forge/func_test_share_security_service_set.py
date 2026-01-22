import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_set(self):
    arglist = [self.security_service.id, '--dns-ip', self.security_service.dns_ip, '--ou', self.security_service.ou, '--server', self.security_service.server, '--domain', self.security_service.domain, '--user', self.security_service.user, '--password', self.security_service.password, '--name', self.security_service.name, '--description', self.security_service.description, '--default-ad-site', self.security_service.default_ad_site]
    verifylist = [('security_service', self.security_service.id), ('dns_ip', self.security_service.dns_ip), ('ou', self.security_service.ou), ('server', self.security_service.server), ('domain', self.security_service.domain), ('user', self.security_service.user), ('password', self.security_service.password), ('name', self.security_service.name), ('description', self.security_service.description), ('default_ad_site', self.security_service.default_ad_site)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.security_service.update.assert_called_with(dns_ip=self.security_service.dns_ip, server=self.security_service.server, domain=self.security_service.domain, user=self.security_service.user, password=self.security_service.password, name=self.security_service.name, description=self.security_service.description, ou=self.security_service.ou, default_ad_site=self.security_service.default_ad_site)
    self.assertIsNone(result)