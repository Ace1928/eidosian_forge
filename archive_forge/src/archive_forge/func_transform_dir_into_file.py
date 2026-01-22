import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def transform_dir_into_file(self, path, content, base=branch_dir):
    osutils.delete_any(osutils.pathjoin(base, path))
    self.set_file_content(path, content, base)
    self.tree.commit('change %s from dir to file' % path)