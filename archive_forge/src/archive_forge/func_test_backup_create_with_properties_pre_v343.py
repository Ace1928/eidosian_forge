from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_create_with_properties_pre_v343(self):
    self._set_mock_microversion('3.42')
    arglist = ['--property', 'foo=bar', '--property', 'wow=much-cool', self.new_backup.volume_id]
    verifylist = [('properties', {'foo': 'bar', 'wow': 'much-cool'}), ('volume', self.new_backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.43 or greater', str(exc))