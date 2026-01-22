import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_inaccessible_implicit(self):
    osutils.normalized_filename = osutils._inaccessible_normalized_filename
    self.assertRaises(errors.InvalidNormalization, self.wt.smart_add, [])