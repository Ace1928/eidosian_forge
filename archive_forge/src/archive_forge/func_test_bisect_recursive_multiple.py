import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_recursive_multiple(self):
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisectRecursive(expected, [b'a', b'b/c'], state, [b'a', b'b/c'])
    self.assertBisectRecursive(expected, [b'b/d', b'b/d/e'], state, [b'b/d', b'b/d/e'])