import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_split(self):
    outer_tree = self.make_branch_and_tree('outer')
    outer_tree.commit('Add root')
    inner_tree = self.make_branch_and_tree('outer/inner')
    rev2 = inner_tree.commit('Add root')
    outer_tree.subsume(inner_tree)
    self.expectFailure('Cannot shelve a join back to the inner tree.', self.assertRaises, AssertionError, self.assertRaises, ValueError, self.shelve_all, outer_tree, rev2)