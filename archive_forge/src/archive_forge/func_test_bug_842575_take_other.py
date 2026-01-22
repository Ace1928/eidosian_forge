from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
def test_bug_842575_take_other(self):
    self.run_script('$ brz init -q trunk\n$ echo original > trunk/foo\n$ brz add -q trunk/foo\n$ brz commit -q -m first trunk\n$ brz checkout -q --lightweight trunk tree\n$ brz rm -q trunk/foo\n$ brz ignore -d trunk foo\n$ brz commit -q -m second trunk\n$ echo modified > tree/foo\n$ brz update tree\n2>+N  .bzrignore\n2>RM  foo => foo.THIS\n2>Contents conflict in foo\n2>1 conflicts encountered.\n2>Updated to revision 2 of branch ...\n$ brz resolve --take-other --all -d tree\n2>1 conflict resolved, 0 remaining\n')
    try:
        self.run_script('$ brz status tree\n$ echo mustignore > tree/foo\n$ brz status tree\n')
    except AssertionError:
        raise KnownFailure('bug 842575')