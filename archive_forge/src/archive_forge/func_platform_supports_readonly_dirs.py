import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def platform_supports_readonly_dirs(self):
    if sys.platform in ('win32', 'cygwin'):
        return False
    return True