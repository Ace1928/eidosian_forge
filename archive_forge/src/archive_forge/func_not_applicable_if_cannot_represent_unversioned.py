import breezy
from breezy import revisiontree, tests
from breezy.bzr import inventorytree
from breezy.bzr.workingtree_3 import WorkingTreeFormat3
from breezy.bzr.workingtree_4 import WorkingTreeFormat4
from breezy.tests import default_transport, multiply_tests
from breezy.tests.per_tree import (TestCaseWithTree, return_parameter,
from breezy.tree import InterTree
def not_applicable_if_cannot_represent_unversioned(self, tree):
    if isinstance(tree, revisiontree.RevisionTree):
        raise tests.TestNotApplicable('cannot represent unversioned files')