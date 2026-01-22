import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def get_unbalanced_tree_pair(self):
    """Return two branches, a and b, with one file in a."""
    tree_a = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/b', b'b')])
    tree_a.add('b')
    tree_a.commit('silly commit')
    tree_b = self.make_branch_and_tree('b')
    return (tree_a, tree_b)