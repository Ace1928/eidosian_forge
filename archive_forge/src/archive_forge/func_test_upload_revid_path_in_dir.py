import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_revid_path_in_dir(self):
    self.make_branch_and_working_tree()
    self.add_dir('dir')
    self.add_file('dir/goodbye', b'baz')
    revid_path = 'dir/revid-path'
    self.tree.branch.get_config_stack().set('upload_revid_location', revid_path)
    self.assertUpPathDoesNotExist(revid_path)
    self.do_full_upload()
    self.add_file('dir/hello', b'foo')
    self.do_upload()
    self.assertUpPathExists(revid_path)
    self.assertUpFileEqual(b'baz', 'dir/goodbye')
    self.assertUpFileEqual(b'foo', 'dir/hello')