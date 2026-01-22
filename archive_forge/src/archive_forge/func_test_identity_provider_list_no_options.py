import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.identity_providers_mock.list.assert_called_with()
    collist = ('ID', 'Enabled', 'Domain ID', 'Description')
    self.assertEqual(collist, columns)
    datalist = ((identity_fakes.idp_id, True, identity_fakes.domain_id, identity_fakes.idp_description),)
    self.assertCountEqual(datalist, tuple(data))