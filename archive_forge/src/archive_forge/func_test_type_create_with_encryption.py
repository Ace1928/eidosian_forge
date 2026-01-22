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
def test_type_create_with_encryption(self):
    encryption_info = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': '128', 'control_location': 'front-end'}
    encryption_type = volume_fakes.create_one_encryption_volume_type(attrs=encryption_info)
    self.new_volume_type = volume_fakes.create_one_volume_type(attrs={'encryption': encryption_info})
    self.volume_types_mock.create.return_value = self.new_volume_type
    self.volume_encryption_types_mock.create.return_value = encryption_type
    encryption_columns = ('description', 'encryption', 'id', 'is_public', 'name')
    encryption_data = (self.new_volume_type.description, format_columns.DictColumn(encryption_info), self.new_volume_type.id, True, self.new_volume_type.name)
    arglist = ['--encryption-provider', 'LuksEncryptor', '--encryption-cipher', 'aes-xts-plain64', '--encryption-key-size', '128', '--encryption-control-location', 'front-end', self.new_volume_type.name]
    verifylist = [('encryption_provider', 'LuksEncryptor'), ('encryption_cipher', 'aes-xts-plain64'), ('encryption_key_size', 128), ('encryption_control_location', 'front-end'), ('name', self.new_volume_type.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=None)
    body = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    self.volume_encryption_types_mock.create.assert_called_with(self.new_volume_type, body)
    self.assertEqual(encryption_columns, columns)
    self.assertCountEqual(encryption_data, data)