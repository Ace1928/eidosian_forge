from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_missing_revisions_unknown_stop_rev(self):
    interrepo = self._get_interrepo()
    interrepo.source_store.lock_read()
    self.addCleanup(interrepo.source_store.unlock)
    self.assertEqual([], list(interrepo.missing_revisions([(None, b'unknown')])))