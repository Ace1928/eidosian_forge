from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
def test_resolve_via_directory_option(self):
    self.run_script('$ brz resolve -d branch myfile\n2>1 conflict resolved, 2 remaining\n')