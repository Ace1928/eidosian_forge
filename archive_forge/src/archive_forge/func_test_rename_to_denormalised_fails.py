import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_to_denormalised_fails(self):
    if osutils.normalizes_filenames():
        raise tests.TestNotApplicable('OSX normalizes filenames')
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    self.assertRaises((errors.InvalidNormalization, UnicodeEncodeError), tree.rename_one, 'a', 'bårry')