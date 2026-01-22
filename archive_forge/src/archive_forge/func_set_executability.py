import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def set_executability(wt, path, executable=True):
    """Set the executable bit for the file at path in the working tree

    os.chmod() doesn't work on windows. But TreeTransform can mark or
    unmark a file as executable.
    """
    with wt.transform() as tt:
        tt.set_executability(executable, tt.trans_id_tree_path(path))
        tt.apply()