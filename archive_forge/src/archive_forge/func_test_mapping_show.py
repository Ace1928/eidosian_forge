import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_mapping_show(self):
    arglist = [identity_fakes.mapping_id]
    verifylist = [('mapping', identity_fakes.mapping_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.mapping_mock.get.assert_called_with(identity_fakes.mapping_id)
    collist = ('id', 'rules')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.mapping_id, identity_fakes.MAPPING_RULES)
    self.assertEqual(datalist, data)