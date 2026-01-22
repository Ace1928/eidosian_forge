from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_pointless_fetch_refs_old_mapping(self):
    interrepo = self._get_interrepo(mapping=BzrGitMappingv1())
    interrepo.fetch_refs(lambda x: {}, lossy=False)