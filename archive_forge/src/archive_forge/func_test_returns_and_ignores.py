import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_returns_and_ignores(self):
    """Correctly returns added/ignored files"""
    wt = self.make_branch_and_tree('.')
    ignores._set_user_ignores(['*.py[co]'])
    self.build_tree(['inertiatic/', 'inertiatic/esp', 'inertiatic/CVS', 'inertiatic/foo.pyc'])
    added, ignored = wt.smart_add('.')
    if wt.has_versioned_directories():
        self.assertSubset(('inertiatic', 'inertiatic/esp', 'inertiatic/CVS'), added)
    else:
        self.assertSubset(('inertiatic/esp', 'inertiatic/CVS'), added)
    self.assertSubset(('*.py[co]',), ignored)
    self.assertSubset(('inertiatic/foo.pyc',), ignored['*.py[co]'])