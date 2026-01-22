import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_parent_map_type(self):
    tree = self.make_branch_and_tree('here')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    rev1 = tree.commit('initial commit')
    rev2 = tree.commit('next commit')
    graph = tree.branch.repository.get_graph()
    parents = graph.get_parent_map([_mod_revision.NULL_REVISION, rev1, rev2])
    for value in parents.values():
        self.assertIsInstance(value, tuple)