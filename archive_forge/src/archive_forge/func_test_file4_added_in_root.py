import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_file4_added_in_root(self):
    outer, inner, revs = self.make_outer_tree()
    nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[3])
    if outer.supports_rename_tracking():
        self.assertEqual(1, len(nb_conflicts))
    else:
        self.assertEqual(0, len(nb_conflicts))
    self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/file3', 'file4', 'foo'], outer)