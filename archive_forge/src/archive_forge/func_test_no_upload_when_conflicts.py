import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_no_upload_when_conflicts(self):
    self.make_branch_and_working_tree()
    self.add_file('a', b'foo')
    self.run_bzr('branch branch other')
    self.modify_file('a', b'bar')
    other_tree = workingtree.WorkingTree.open('other')
    self.set_file_content('a', b'baz', 'other/')
    other_tree.commit('modify file a')
    self.run_bzr('merge -d branch other', retcode=1)
    self.assertRaises(errors.UncommittedChanges, self.do_upload)