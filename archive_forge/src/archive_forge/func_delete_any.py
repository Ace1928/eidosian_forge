import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def delete_any(self, path, base=branch_dir):
    self.tree.remove([path], keep_files=False)
    self.tree.commit('delete %s' % path)