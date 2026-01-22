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
def preview_tree_post(testcase, tree):
    basis = tree.basis_tree()
    tt = basis.preview_transform()
    testcase.addCleanup(tt.finalize)
    tree.lock_read()
    testcase.addCleanup(tree.unlock)
    pp = None
    es = contextlib.ExitStack()
    testcase.addCleanup(es.close)
    transform._prepare_revert_transform(es, basis, tree, tt, None, False, None, basis, {})
    preview_tree = tt.get_preview_tree()
    preview_tree.set_parent_ids(tree.get_parent_ids())
    return preview_tree