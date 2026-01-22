import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def myhash(val):
    """This the hash used by RenameMap."""
    return hash(val) % (1024 * 1024 * 10)