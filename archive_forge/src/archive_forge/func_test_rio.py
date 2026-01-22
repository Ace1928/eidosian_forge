import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_rio(self):
    """Test that we can pass --format=rio"""
    self.create_tree()
    txt = self.run_bzr('version-info branch')[0]
    txt1 = self.run_bzr('version-info --format rio branch')[0]
    txt2 = self.run_bzr('version-info --format=rio branch')[0]
    self.assertEqualNoBuildDate(txt, txt1)
    self.assertEqualNoBuildDate(txt, txt2)