import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
def test_qos_create_without_properties(self):
    arglist = [self.new_qos_spec.name]
    verifylist = [('name', self.new_qos_spec.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.qos_mock.create.assert_called_with(self.new_qos_spec.name, {'consumer': 'both'})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)