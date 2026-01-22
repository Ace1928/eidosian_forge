import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
def get_tree_no_parents_abc_content_6(self, tree, converter=None):
    """return a test tree with a, b/, e contents.

        This variation renames b/c to e, and makes it executable.
        """
    self._make_abc_tree(tree)
    tt = tree.transform()
    trans_id = tt.trans_id_tree_path('b/c')
    parent_trans_id = tt.trans_id_tree_path('')
    tt.adjust_path('e', parent_trans_id, trans_id)
    tt.set_executability(True, trans_id)
    tt.apply()
    return self._convert_tree(tree, converter)