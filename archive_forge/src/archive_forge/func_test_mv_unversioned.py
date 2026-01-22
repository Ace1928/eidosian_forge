import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_unversioned(self):
    self.build_tree(['unversioned.txt'])
    self.run_bzr_error(['^brz: ERROR: Could not rename unversioned.txt => elsewhere. .*unversioned.txt is not versioned\\.$'], 'mv unversioned.txt elsewhere')