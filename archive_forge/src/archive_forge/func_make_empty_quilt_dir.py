import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def make_empty_quilt_dir(self, path):
    source = self.make_branch_and_tree(path)
    self.build_tree([os.path.join(path, n) for n in ['patches/']])
    self.build_tree_contents([(os.path.join(path, 'patches/series'), '\n')])
    source.add(['patches', 'patches/series'])
    return source