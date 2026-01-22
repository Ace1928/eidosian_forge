import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def transform_file_into_dir(self, path, base=branch_dir):
    self.tree.remove([path], keep_files=False)
    os.mkdir(osutils.pathjoin(base, path))
    self.tree.add(path)
    self.tree.commit('change %s from file to dir' % path)