import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_page_size_too_small(self):
    """If the page size is too small, we will auto increase it."""
    tree, state, expected = self.create_basic_dirstate()
    state._bisect_page_size = 50
    self.assertBisect(expected, [None], state, [b'b/e'])
    self.assertBisect(expected, [[b'a']], state, [b'a'])
    self.assertBisect(expected, [[b'b']], state, [b'b'])
    self.assertBisect(expected, [[b'b/c']], state, [b'b/c'])
    self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
    self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
    self.assertBisect(expected, [[b'b-c']], state, [b'b-c'])
    self.assertBisect(expected, [[b'f']], state, [b'f'])