import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def modify_symlink(self, path, target, base=branch_dir):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    full_path = osutils.pathjoin(base, path)
    os.unlink(full_path)
    os.symlink(target, full_path)
    self.tree.commit('modify symlink {} -> {}'.format(path, target))