import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_show(self):
    arglist = [self.security_service.id]
    verifylist = [('security_service', self.security_service.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.security_services_mock.get.assert_called_with(self.security_service.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)