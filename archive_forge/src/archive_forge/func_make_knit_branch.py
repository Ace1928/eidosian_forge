from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def make_knit_branch(self, relpath):
    old_bdf = controldir.format_registry.make_controldir('knit')
    return controldir.ControlDir.create_branch_convenience(relpath, format=old_bdf)