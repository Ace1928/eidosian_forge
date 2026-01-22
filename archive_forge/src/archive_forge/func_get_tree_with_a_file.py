import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def get_tree_with_a_file(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a file'])
    tree.add('a file', ids=b'a-file-id')
    return tree