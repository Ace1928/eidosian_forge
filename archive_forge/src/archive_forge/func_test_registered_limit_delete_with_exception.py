import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_registered_limit_delete_with_exception(self):
    return_value = ksa_exceptions.NotFound()
    self.registered_limit_mock.delete.side_effect = return_value
    arglist = ['fake-registered-limit-id']
    verifylist = [('registered_limit_id', ['fake-registered-limit-id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 1 registered limits failed to delete.', str(e))