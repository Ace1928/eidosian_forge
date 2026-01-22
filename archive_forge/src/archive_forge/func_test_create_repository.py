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
def test_create_repository(self):
    if not self.bzrdir_format.is_supported():
        return
    t = self.get_transport()
    made_control = self.bzrdir_format.initialize(t.base)
    made_repo = made_control.create_repository()
    made_repo.has_revision(b'foo')
    self.assertEqual(made_control, made_repo.controldir)