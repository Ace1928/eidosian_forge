import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_shows_pull_with_location(self):
    wt = self.create_simple_tree()
    script = ScriptRunner()
    script.run_script(self, '\n$ brz uncommit --force tree\n    2 ...\n      second commit\n...\nThe above revision(s) will be removed.\nYou can restore the old tip by running:\n  brz pull -d tree tree -r revid:a2\n')