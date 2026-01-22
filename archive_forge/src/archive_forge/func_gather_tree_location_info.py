import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def gather_tree_location_info(self, tree):
    return info.gather_location_info(tree.branch.repository, tree.branch, tree, tree.controldir)