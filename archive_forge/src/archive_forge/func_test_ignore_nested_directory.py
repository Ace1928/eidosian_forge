import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_ignore_nested_directory(self):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    self.add_file('.bzrignore-upload', b'dir')
    self.add_dir('dir')
    self.add_dir('dir/foo')
    self.add_file('dir/foo/bar', b'bar contents')
    self.do_upload()
    self.assertUpPathDoesNotExist('dir')
    self.assertUpPathDoesNotExist('dir/foo/bar')