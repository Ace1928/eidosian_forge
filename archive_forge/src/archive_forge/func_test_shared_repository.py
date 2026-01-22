import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_shared_repository(self):
    t = get_transport('.')
    self.assertRaises(errors.SharedRepositoriesUnsupported, dir.LocalGitControlDirFormat().initialize_on_transport_ex, t, shared_repo=True)