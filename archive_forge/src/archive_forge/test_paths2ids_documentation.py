from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
Check the basic nasty corner case that path2ids should handle.

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
        