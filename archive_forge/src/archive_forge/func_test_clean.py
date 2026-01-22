import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_clean(self):
    """Test that --check-clean includes the right info"""
    self.create_tree()
    txt = self.run_bzr('version-info branch --check-clean')[0]
    self.assertContainsRe(txt, 'clean: True')
    self.build_tree_contents([('branch/c', b'now unclean\n')])
    txt = self.run_bzr('version-info branch --check-clean')[0]
    self.assertContainsRe(txt, 'clean: False')
    txt = self.run_bzr('version-info branch --check-clean --include-file-revisions')[0]
    self.assertContainsRe(txt, 'revision: unversioned')
    os.remove('branch/c')