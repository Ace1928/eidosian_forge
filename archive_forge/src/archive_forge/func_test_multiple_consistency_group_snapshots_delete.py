from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_multiple_consistency_group_snapshots_delete(self):
    arglist = []
    for c in self.consistency_group_snapshots:
        arglist.append(c.id)
    verifylist = [('consistency_group_snapshot', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for c in self.consistency_group_snapshots:
        calls.append(call(c.id))
    self.cgsnapshots_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)