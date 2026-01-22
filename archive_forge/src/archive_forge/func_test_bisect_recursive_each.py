import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_recursive_each(self):
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisectRecursive(expected, [b'a'], state, [b'a'])
    self.assertBisectRecursive(expected, [b'b/c'], state, [b'b/c'])
    self.assertBisectRecursive(expected, [b'b/d/e'], state, [b'b/d/e'])
    self.assertBisectRecursive(expected, [b'b-c'], state, [b'b-c'])
    self.assertBisectRecursive(expected, [b'b/d', b'b/d/e'], state, [b'b/d'])
    self.assertBisectRecursive(expected, [b'b', b'b/c', b'b/d', b'b/d/e'], state, [b'b'])
    self.assertBisectRecursive(expected, [b'', b'a', b'b', b'b-c', b'f', b'b/c', b'b/d', b'b/d/e'], state, [b''])