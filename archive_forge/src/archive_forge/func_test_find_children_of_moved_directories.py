from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_find_children_of_moved_directories(self):
    """Check the basic nasty corner case that path2ids should handle.

        This is the following situation:
        basis:
          / ROOT
          /dir dir
          /dir/child-moves child-moves
          /dir/child-stays child-stays
          /dir/child-goes  child-goes

        current tree:
          / ROOT
          /child-moves child-moves
          /newdir newdir
          /newdir/dir  dir
          /newdir/dir/child-stays child-stays
          /newdir/dir/new-child   new-child

        In english: we move a directory under a directory that was a sibling,
        and at the same time remove, or move out of the directory, some of its
        children, and give it a new child previous absent or a sibling.

        current_tree.path2ids(['newdir'], [basis]) is meant to handle this
        correctly: that is it should return the ids:
          newdir because it was provided
          dir, because its under newdir in current
          child-moves because its under dir in old
          child-stays either because its under newdir/dir in current, or under dir in old
          child-goes because its under dir in old.
          new-child because its under dir in new

        Symmetrically, current_tree.path2ids(['dir'], [basis]) is meant to show
        new-child, even though its not under the path 'dir' in current, because
        its under a path selected by 'dir' in basis:
          dir because its selected in basis.
          child-moves because its under dir in old
          child-stays either because its under newdir/dir in current, or under dir in old
          child-goes because its under dir in old.
          new-child because its under dir in new.
        """
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryTree):
        raise TestNotApplicable('test not applicable on non-inventory tests')
    self.build_tree(['tree/dir/', 'tree/dir/child-moves', 'tree/dir/child-stays', 'tree/dir/child-goes'])
    tree.add(['dir', 'dir/child-moves', 'dir/child-stays', 'dir/child-goes'], ids=[b'dir', b'child-moves', b'child-stays', b'child-goes'])
    tree.commit('create basis')
    basis = tree.basis_tree()
    tree.unversion(['dir/child-goes'])
    tree.rename_one('dir/child-moves', 'child-moves')
    self.build_tree(['tree/newdir/'])
    tree.add(['newdir'], ids=[b'newdir'])
    tree.rename_one('dir/child-stays', 'child-stays')
    tree.rename_one('dir', 'newdir/dir')
    tree.rename_one('child-stays', 'newdir/dir/child-stays')
    self.build_tree(['tree/newdir/dir/new-child'])
    tree.add(['newdir/dir/new-child'], ids=[b'new-child'])
    self.assertExpectedIds([b'newdir', b'dir', b'child-moves', b'child-stays', b'child-goes', b'new-child'], tree, ['newdir'], [basis])
    self.assertExpectedIds([b'dir', b'child-moves', b'child-stays', b'child-goes', b'new-child'], tree, ['dir'], [basis])