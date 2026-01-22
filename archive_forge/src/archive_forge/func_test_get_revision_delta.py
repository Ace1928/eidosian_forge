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
def test_get_revision_delta(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/foo'])
    tree_a.add('foo')
    rev1 = tree_a.commit('rev1')
    self.build_tree(['a/vla'])
    tree_a.add('vla')
    rev2 = tree_a.commit('rev2')
    delta = tree_a.branch.repository.get_revision_delta(rev1)
    self.assertIsInstance(delta, _mod_delta.TreeDelta)
    self.assertEqual([('foo', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])
    delta = tree_a.branch.repository.get_revision_delta(rev2)
    self.assertIsInstance(delta, _mod_delta.TreeDelta)
    self.assertEqual([('vla', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])