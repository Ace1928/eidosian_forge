from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_create_network_options(self, sgr_mock):
    arglist = ['--ingress', '--ethertype', 'IPv4', '--icmp-type', '3', '--icmp-code', '11', '--project', self.project.name, '--project-domain', self.domain.name, self._security_group['id']]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])