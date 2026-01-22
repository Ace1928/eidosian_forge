from breezy.revision import NULL_REVISION
from breezy.tests.per_repository import TestCaseWithRepository
def test_NULL(self):
    repo = self.make_repository('.')
    self.assertEqual({NULL_REVISION}, repo.has_revisions([NULL_REVISION]))