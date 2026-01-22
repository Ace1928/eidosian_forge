from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_new_in_other_floats_to_top(self):
    """Changes at the top of 'other' float to the top.

        Given a changelog in THIS containing::

          NEW-1
          OLD-1

        and a changelog in OTHER containing::

          NEW-2
          OLD-1

        it will merge as::

          NEW-2
          NEW-1
          OLD-1
        """
    base_entries = [b'OLD-1']
    this_entries = [b'NEW-1', b'OLD-1']
    other_entries = [b'NEW-2', b'OLD-1']
    result_entries = changelog_merge.merge_entries(base_entries, this_entries, other_entries)
    self.assertEqual([b'NEW-2', b'NEW-1', b'OLD-1'], result_entries)