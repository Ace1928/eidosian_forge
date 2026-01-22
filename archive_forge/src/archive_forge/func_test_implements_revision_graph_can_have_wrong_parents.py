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
def test_implements_revision_graph_can_have_wrong_parents(self):
    """All repositories should implement
        revision_graph_can_have_wrong_parents, so that check and reconcile can
        work correctly.
        """
    repo = self.make_repository('.')
    if not repo._format.revision_graph_can_have_wrong_parents:
        return
    repo.lock_read()
    self.addCleanup(repo.unlock)
    list(repo._find_inconsistent_revision_parents())
    repo._check_for_inconsistent_revision_parents()