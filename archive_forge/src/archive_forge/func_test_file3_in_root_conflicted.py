import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_file3_in_root_conflicted(self):
    outer, inner, revs = self.make_outer_tree()
    outer.remove(['dir-outer/file3'], keep_files=False)
    outer.commit('delete file3')
    nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[2])
    if outer.supports_rename_tracking():
        self.assertEqual(4, len(nb_conflicts))
    else:
        self.assertEqual(1, len(nb_conflicts))
    self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'file3.BASE', 'file3.OTHER', 'foo'], outer)