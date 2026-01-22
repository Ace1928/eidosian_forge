from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_lightweight_checkout(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    for count in range(9):
        t.commit(message='commit %d' % count)
    self.reset_smart_call_log()
    out, err = self.run_bzr(['checkout', '--lightweight', self.get_url('from'), 'target'])
    self.assertLength(13, self.hpss_calls)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)