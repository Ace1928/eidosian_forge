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
def test_clone_to_default_format(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/foo'])
    tree_a.add('foo')
    rev1 = tree_a.commit('rev1')
    bzrdirb = self.make_controldir('b')
    repo_b = tree_a.branch.repository.clone(bzrdirb)
    tree_b = repo_b.revision_tree(rev1)
    tree_b.lock_read()
    self.addCleanup(tree_b.unlock)
    tree_b.get_file_text('foo')
    repo_b.get_revision(rev1)