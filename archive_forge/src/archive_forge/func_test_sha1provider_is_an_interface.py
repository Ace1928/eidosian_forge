import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_sha1provider_is_an_interface(self):
    p = dirstate.SHA1Provider()
    self.assertRaises(NotImplementedError, p.sha1, 'foo')
    self.assertRaises(NotImplementedError, p.stat_and_sha1, 'foo')