from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_push_smart_stacked_streaming_acceptance(self):
    self.setup_smart_server_with_call_log()
    parent = self.make_branch_and_tree('parent', format='1.9')
    parent.commit(message='first commit')
    local = parent.controldir.sprout('local').open_workingtree()
    local.commit(message='local commit')
    self.reset_smart_call_log()
    self.run_bzr(['push', '--stacked', '--stacked-on', '../parent', self.get_url('public')], working_dir='local')
    self.assertLength(15, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
    remote = branch.Branch.open('public')
    self.assertEndsWith(remote.get_stacked_on_url(), '/parent')