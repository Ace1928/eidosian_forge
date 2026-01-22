import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_list_ou_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.43')
    arglist = ['--ou', self.services_list[0].ou]
    verifylist = [('ou', self.services_list[0].ou)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)