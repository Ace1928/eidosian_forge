import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_get_head_branch_reference(self):
    GitRepo.init('.')
    gd = controldir.ControlDir.open('.')
    self.assertEqual('%s,branch=master' % urlutils.local_path_to_url(os.path.abspath('.')), gd.get_branch_reference())