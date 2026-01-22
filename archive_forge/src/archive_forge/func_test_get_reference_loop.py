import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_get_reference_loop(self):
    r = GitRepo.init('.')
    r.refs.set_symbolic_ref(b'refs/heads/loop', b'refs/heads/loop')
    gd = controldir.ControlDir.open('.')
    self.assertRaises(controldir.BranchReferenceLoop, gd.get_branch_reference, name='loop')