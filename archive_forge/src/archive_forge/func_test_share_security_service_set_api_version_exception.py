import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data('2.43', '2.75')
def test_share_security_service_set_api_version_exception(self, version):
    self.app.client_manager.share.api_version = api_versions.APIVersion(version)
    arglist = [self.security_service.id]
    verifylist = [('security_service', self.security_service.id)]
    if api_versions.APIVersion(version) <= api_versions.APIVersion('2.43'):
        arglist.extend(['--ou', self.security_service.ou])
        verifylist.append(('ou', self.security_service.ou))
    if api_versions.APIVersion(version) <= api_versions.APIVersion('2.75'):
        arglist.extend(['--default-ad-site', self.security_service.default_ad_site])
        verifylist.append(('default_ad_site', self.security_service.default_ad_site))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)