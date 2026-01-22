from ... import branch, builtins
from .. import transport_util, ui_testing
def test_pull_with_bound_branch(self):
    master_wt = self.make_branch_and_tree('master')
    local_wt = self.make_branch_and_tree('local')
    master_branch = branch.Branch.open(self.get_url('master'))
    local_wt.branch.bind(master_branch)
    remote_wt = self.make_branch_and_tree('remote')
    remote_wt.commit('empty commit')
    self.start_logging_connections()
    pull = builtins.cmd_pull()
    pull.outf = ui_testing.StringIOWithEncoding()
    pull.run(self.get_url('remote'), directory='local')
    self.assertEqual(2, len(self.connections))