from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_consistency_group_snapshot_delete(self):
    arglist = [self.consistency_group_snapshots[0].id]
    verifylist = [('consistency_group_snapshot', [self.consistency_group_snapshots[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.cgsnapshots_mock.delete.assert_called_once_with(self.consistency_group_snapshots[0].id)
    self.assertIsNone(result)