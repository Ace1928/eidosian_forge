import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_interactive(self):
    """Uncommit seeks confirmation, and doesn't proceed without it."""
    wt = self.create_simple_tree()
    os.chdir('tree')
    run_script(self, '\n        $ brz uncommit\n        ...\n        The above revision(s) will be removed.\n        2>Uncommit these revisions? ([y]es, [n]o): no\n        <n\n        Canceled\n        ')
    self.assertEqual([b'a2'], wt.get_parent_ids())