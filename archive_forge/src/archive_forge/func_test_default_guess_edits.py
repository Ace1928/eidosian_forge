from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_default_guess_edits(self):
    """default_guess_edits matches a new entry only once.

        (Even when that entry is the best match for multiple old entries.)
        """
    new_in_other = [(b'AAAAA',), (b'BBBBB',)]
    deleted_in_other = [(b'DDDDD',), (b'BBBBBx',), (b'BBBBBxx',)]
    result = changelog_merge.default_guess_edits(new_in_other, deleted_in_other)
    self.assertEqual(([(b'AAAAA',)], [(b'DDDDD',), (b'BBBBBxx',)], [((b'BBBBBx',), (b'BBBBB',))]), result)