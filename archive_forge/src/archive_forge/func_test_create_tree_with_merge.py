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
def test_create_tree_with_merge(self):
    tree, revmap = self.create_tree_with_merge()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    graph = tree.branch.repository.get_graph()
    ancestry_graph = graph.get_parent_map(tree.branch.repository.all_revision_ids())
    self.assertEqual({revmap['1']: (b'null:',), revmap['2']: (revmap['1'],), revmap['1.1.1']: (revmap['1'],), revmap['3']: (revmap['2'], revmap['1.1.1'])}, ancestry_graph)