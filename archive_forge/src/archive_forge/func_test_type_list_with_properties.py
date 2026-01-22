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
def test_type_list_with_properties(self):
    self.app.client_manager.volume.api_version = api_versions.APIVersion('3.52')
    arglist = ['--property', 'foo=bar', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1']
    verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', False), ('properties', {'foo': 'bar'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.list.assert_called_once_with(search_opts={'extra_specs': {'foo': 'bar', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'}}, is_public=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))