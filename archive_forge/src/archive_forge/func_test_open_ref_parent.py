import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_open_ref_parent(self):
    r = GitRepo.init('.')
    cid = r.do_commit(message=b'message', ref=b'refs/heads/foo/bar')
    gd = controldir.ControlDir.open('.')
    self.assertRaises(errors.NotBranchError, gd.open_branch, 'foo')