import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
def test_export_nested_nonrecurse(self):
    self.prepare_nested_export(False)
    names = self.get_export_names()
    self.assertNotIn('output/subdir/b', names)