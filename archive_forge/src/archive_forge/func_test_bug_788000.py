from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
def test_bug_788000(self):
    self.run_script('$ brz init a\n$ mkdir a/dir\n$ echo foo > a/dir/file\n$ brz add a/dir\n$ cd a\n$ brz commit -m one\n$ cd ..\n$ brz branch a b\n$ echo bar > b/dir/file\n$ cd a\n$ rm -r dir\n$ brz commit -m two\n$ cd ../b\n', null_output_matches_anything=True)
    self.run_script("$ brz pull\nUsing saved parent location:...\nNow on revision 2.\n2>RM  dir/file => dir/file.THIS\n2>Conflict: can't delete dir because it is not empty.  Not deleting.\n2>Conflict because dir is not versioned, but has versioned children...\n2>Contents conflict in dir/file\n2>3 conflicts encountered.\n")
    self.run_script('$ brz resolve --take-other\n2>deleted dir/file.THIS\n2>deleted dir\n2>3 conflicts resolved, 0 remaining\n')