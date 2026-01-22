import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_skip_nested_trees(self):
    """Test smart-adding a nested tree ignors it and warns."""
    wt = self.make_branch_and_tree('.')
    nested_wt = self.make_branch_and_tree('nested')
    warnings = []

    def warning(*args):
        warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    wt.smart_add(('.',))
    self.assertFalse(wt.is_versioned('nested'))
    self.assertEqual(['skipping nested tree %r' % nested_wt.basedir], warnings)