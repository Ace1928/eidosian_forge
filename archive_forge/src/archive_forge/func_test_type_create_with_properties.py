from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
def test_type_create_with_properties(self):
    arglist = ['--property', 'myprop=myvalue', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1', self.new_volume_type.name]
    verifylist = [('properties', {'myprop': 'myvalue'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1']), ('name', self.new_volume_type.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=None)
    self.new_volume_type.set_keys.assert_called_once_with({'myprop': 'myvalue', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'})
    self.columns += ('properties',)
    self.data += (format_columns.DictColumn(None),)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)