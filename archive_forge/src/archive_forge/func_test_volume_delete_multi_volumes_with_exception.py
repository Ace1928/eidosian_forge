from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
def test_volume_delete_multi_volumes_with_exception(self):
    volumes = self.setup_volumes_mock(count=2)
    arglist = [volumes[0].id, 'unexist_volume']
    verifylist = [('force', False), ('purge', False), ('volumes', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [volumes[0], exceptions.CommandError]
    with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 volumes failed to delete.', str(e))
        find_mock.assert_any_call(self.volumes_mock, volumes[0].id)
        find_mock.assert_any_call(self.volumes_mock, 'unexist_volume')
        self.assertEqual(2, find_mock.call_count)
        self.volumes_mock.delete.assert_called_once_with(volumes[0].id, cascade=False)