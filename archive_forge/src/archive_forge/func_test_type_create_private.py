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
def test_type_create_private(self):
    arglist = ['--description', self.new_volume_type.description, '--private', '--project', self.project.id, self.new_volume_type.name]
    verifylist = [('description', self.new_volume_type.description), ('is_public', False), ('project', self.project.id), ('name', self.new_volume_type.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=self.new_volume_type.description, is_public=False)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)