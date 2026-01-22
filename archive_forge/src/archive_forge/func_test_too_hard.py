from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_too_hard(self):
    """A conflict this plugin cannot resolve raises EntryConflict.
        """
    self.assertRaises(changelog_merge.EntryConflict, changelog_merge.merge_entries, [(entry,) for entry in sample2_base_entries], [], [(entry,) for entry in sample2_other_entries])