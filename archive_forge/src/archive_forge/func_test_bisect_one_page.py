import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_one_page(self):
    """Test bisect when there is only 1 page to read"""
    tree, state, expected = self.create_basic_dirstate()
    state._bisect_page_size = 5000
    self.assertBisect(expected, [[b'']], state, [b''])
    self.assertBisect(expected, [[b'a']], state, [b'a'])
    self.assertBisect(expected, [[b'b']], state, [b'b'])
    self.assertBisect(expected, [[b'b/c']], state, [b'b/c'])
    self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
    self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
    self.assertBisect(expected, [[b'b-c']], state, [b'b-c'])
    self.assertBisect(expected, [[b'f']], state, [b'f'])
    self.assertBisect(expected, [[b'a'], [b'b'], [b'f']], state, [b'a', b'b', b'f'])
    self.assertBisect(expected, [[b'b/d'], [b'b/d/e'], [b'f']], state, [b'b/d', b'b/d/e', b'f'])
    self.assertBisect(expected, [[b'b'], [b'b/c'], [b'b-c']], state, [b'b', b'b/c', b'b-c'])