import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_nonexisting(self):
    self.run_bzr_error(['^brz: ERROR: Could not copy .*doesnotexist => .*somewhereelse. .*doesnotexist is not versioned\\.$'], 'cp doesnotexist somewhereelse')