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
def get_tree_no_parents_abc_content_2(self, tree, converter=None):
    """return a test tree with a, b/, b/c contents.

        This variation changes the content of 'a' to foobar
.
        """
    self._make_abc_tree(tree)
    with open(tree.basedir + '/a', 'wb') as f:
        f.write(b'foobar\n')
    return self._convert_tree(tree, converter)