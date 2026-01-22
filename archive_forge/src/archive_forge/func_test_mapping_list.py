import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_mapping_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.mapping_mock.list.assert_called_with()
    collist = ('ID', 'schema_version')
    self.assertEqual(collist, columns)
    datalist = [(identity_fakes.mapping_id, '1.0'), ('extra_mapping', '2.0')]
    self.assertEqual(datalist, data)