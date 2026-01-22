import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def test_applied_empty(self):
    source = self.make_empty_quilt_dir('source')
    self.build_tree_contents([('source/patches/series', 'patch1.diff\n'), ('source/patches/patch1.diff', 'foob ar')])
    self.assertEqual([], quilt_applied(source))