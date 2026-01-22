import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_recursive_missing(self):
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisectRecursive(expected, [], state, [b'd'])
    self.assertBisectRecursive(expected, [], state, [b'b/e'])
    self.assertBisectRecursive(expected, [], state, [b'g'])
    self.assertBisectRecursive(expected, [b'a'], state, [b'a', b'g'])