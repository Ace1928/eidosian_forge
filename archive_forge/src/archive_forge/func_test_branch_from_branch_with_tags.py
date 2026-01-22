from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_from_branch_with_tags(self):
    self.setup_smart_server_with_call_log()
    builder = self.make_branch_builder('source')
    source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    source.get_config_stack().set('branch.fetch_tags', True)
    source.tags.set_tag('tag-a', rev2)
    source.tags.set_tag('tag-missing', b'missing-rev')
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', self.get_url('source'), 'target'])
    self.assertLength(11, self.hpss_calls)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
    self.assertLength(1, self.hpss_connections)