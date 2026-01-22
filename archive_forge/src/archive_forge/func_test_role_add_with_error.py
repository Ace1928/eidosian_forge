import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_add_with_error(self):
    arglist = [identity_fakes.role_name]
    verifylist = [('user', None), ('group', None), ('domain', None), ('project', None), ('role', identity_fakes.role_name), ('inherited', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)