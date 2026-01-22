from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_create_from_src_source_group_group_snapshot(self):
    self.volume_client.api_version = api_versions.APIVersion('3.14')
    arglist = ['--source-group', self.fake_volume_group.id, '--group-snapshot', self.fake_volume_group_snapshot.id]
    verifylist = [('source_group', self.fake_volume_group.id), ('group_snapshot', self.fake_volume_group_snapshot.id)]
    exc = self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.assertIn('--group-snapshot: not allowed with argument --source-group', str(exc))