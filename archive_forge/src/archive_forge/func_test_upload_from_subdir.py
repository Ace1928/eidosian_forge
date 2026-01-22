import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_from_subdir(self):
    self.make_branch_and_working_tree()
    self.build_tree(['branch/foo/', 'branch/foo/bar'])
    self.tree.add(['foo/', 'foo/bar'])
    self.tree.commit('Add directory')
    self.do_full_upload(directory='branch/foo')