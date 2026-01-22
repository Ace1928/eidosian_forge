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
def test_find_branches_using_standalone(self):
    branch = self.make_branch('branch')
    if not branch.repository._format.supports_nesting_repositories:
        raise tests.TestNotApplicable('format does not support nesting repositories')
    contained = self.make_branch('branch/contained')
    branches = branch.repository.find_branches(using=True)
    self.assertEqual([branch.base], [b.base for b in branches])
    branches = branch.repository.find_branches(using=False)
    self.assertEqual([branch.base, contained.base], [b.base for b in branches])