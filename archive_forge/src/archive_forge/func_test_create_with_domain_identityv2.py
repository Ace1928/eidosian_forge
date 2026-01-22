import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_with_domain_identityv2(self):
    arglist = ['--project', self.project.name, '--project-domain', 'domain-name', self._network.name]
    verifylist = [('enable', True), ('share', None), ('project', self.project.name), ('project_domain', 'domain-name'), ('name', self._network.name), ('external', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(AttributeError, self.cmd.take_action, parsed_args)