import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_show(self):
    arglist = [identity_fakes.role_name]
    verifylist = [('role', identity_fakes.role_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.roles_mock.get.assert_called_with(identity_fakes.role_name)
    collist = ('domain', 'id', 'name')
    self.assertEqual(collist, columns)
    datalist = (None, identity_fakes.role_id, identity_fakes.role_name)
    self.assertEqual(datalist, data)