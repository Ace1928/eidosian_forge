from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_pointless_fetch_refs(self):
    interrepo = self._get_interrepo(mapping=BzrGitMappingExperimental())
    revidmap, old_refs, new_refs = interrepo.fetch_refs(lambda x: {}, lossy=False)
    self.assertEqual(old_refs, {b'HEAD': (b'ref: refs/heads/master', None)})
    self.assertEqual(new_refs, {})