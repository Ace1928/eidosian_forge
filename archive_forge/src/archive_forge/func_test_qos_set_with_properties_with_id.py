import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
def test_qos_set_with_properties_with_id(self):
    arglist = ['--no-property', '--property', 'a=b', '--property', 'c=d', self.qos_spec.id]
    new_property = {'a': 'b', 'c': 'd'}
    verifylist = [('no_property', True), ('property', new_property), ('qos_spec', self.qos_spec.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.qos_mock.unset_keys.assert_called_with(self.qos_spec.id, list(self.qos_spec.specs.keys()))
    self.qos_mock.set_keys.assert_called_with(self.qos_spec.id, {'a': 'b', 'c': 'd'})
    self.assertIsNone(result)