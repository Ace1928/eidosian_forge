import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_delete(self):
    arglist = [self.security_services[0].id, self.security_services[1].id]
    verifylist = [('security_service', [self.security_services[0].id, self.security_services[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.security_services_mock.delete.call_count, len(self.security_services))
    self.assertIsNone(result)