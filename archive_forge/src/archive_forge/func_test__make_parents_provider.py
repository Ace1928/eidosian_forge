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
def test__make_parents_provider(self):
    """Repositories must have a _make_parents_provider method that returns
        an object with a get_parent_map method.
        """
    repo = self.make_repository('repo')
    repo._make_parents_provider().get_parent_map