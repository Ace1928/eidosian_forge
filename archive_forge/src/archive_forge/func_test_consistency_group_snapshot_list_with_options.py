from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_consistency_group_snapshot_list_with_options(self):
    arglist = ['--all-project', '--status', self.consistency_group_snapshots[0].status, '--consistency-group', self.consistency_group.id]
    verifylist = [('all_projects', True), ('long', False), ('status', self.consistency_group_snapshots[0].status), ('consistency_group', self.consistency_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    search_opts = {'all_tenants': True, 'status': self.consistency_group_snapshots[0].status, 'consistencygroup_id': self.consistency_group.id}
    self.consistencygroups_mock.get.assert_called_once_with(self.consistency_group.id)
    self.cgsnapshots_mock.list.assert_called_once_with(detailed=True, search_opts=search_opts)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))