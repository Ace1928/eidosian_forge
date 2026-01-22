from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def skip_if_no_reference(self, tree):
    if not tree.supports_tree_reference():
        raise tests.TestNotApplicable('Tree references not supported')