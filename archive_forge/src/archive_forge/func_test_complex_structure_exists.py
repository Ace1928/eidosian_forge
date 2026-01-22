import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_complex_structure_exists(self):
    state = self.create_complex_dirstate()
    self.addCleanup(state.unlock)
    self.assertEntryEqual(b'', b'', b'a-root-value', state, b'', 0)
    self.assertEntryEqual(b'', b'a', b'a-dir', state, b'a', 0)
    self.assertEntryEqual(b'', b'b', b'b-dir', state, b'b', 0)
    self.assertEntryEqual(b'', b'c', b'c-file', state, b'c', 0)
    self.assertEntryEqual(b'', b'd', b'd-file', state, b'd', 0)
    self.assertEntryEqual(b'a', b'e', b'e-dir', state, b'a/e', 0)
    self.assertEntryEqual(b'a', b'f', b'f-file', state, b'a/f', 0)
    self.assertEntryEqual(b'b', b'g', b'g-file', state, b'b/g', 0)
    self.assertEntryEqual(b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file', state, b'b/h\xc3\xa5', 0)