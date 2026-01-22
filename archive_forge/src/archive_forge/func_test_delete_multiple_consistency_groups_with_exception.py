from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_delete_multiple_consistency_groups_with_exception(self):
    arglist = [self.consistency_groups[0].id, 'unexist_consistency_group']
    verifylist = [('consistency_groups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self.consistency_groups[0], exceptions.CommandError]
    with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 consistency groups failed to delete.', str(e))
        find_mock.assert_any_call(self.consistencygroups_mock, self.consistency_groups[0].id)
        find_mock.assert_any_call(self.consistencygroups_mock, 'unexist_consistency_group')
        self.assertEqual(2, find_mock.call_count)
        self.consistencygroups_mock.delete.assert_called_once_with(self.consistency_groups[0].id, False)