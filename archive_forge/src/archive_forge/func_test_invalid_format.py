import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_invalid_format(self):
    self.run_bzr('version-info --format quijibo', retcode=3)