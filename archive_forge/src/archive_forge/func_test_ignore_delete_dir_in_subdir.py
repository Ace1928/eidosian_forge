import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_ignore_delete_dir_in_subdir(self):
    self.make_branch_and_working_tree()
    self.add_dir('dir')
    self.add_dir('dir/subdir')
    self.add_file('dir/subdir/a', b'foo')
    self.do_full_upload()
    self.add_file('.bzrignore-upload', b'dir/subdir')
    self.rename_any('dir/subdir/a', 'dir/a')
    self.delete_any('dir/subdir')
    self.assertUpFileEqual(b'foo', 'dir/subdir/a')
    self.do_upload()
    self.assertUpFileEqual(b'foo', 'dir/a')