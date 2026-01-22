import os
from breezy import branch, controldir, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import bzrdir
from breezy.bzr.knitrepo import RepositoryFormatKnit1
from breezy.tests import fixtures, test_server
from breezy.tests.blackbox import test_switch
from breezy.tests.features import HardlinkFeature
from breezy.tests.script import run_script
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.urlutils import local_path_to_url, strip_trailing_slash
from breezy.workingtree import WorkingTree
def test_branch_bind(self):
    self.example_branch('a')
    out, err = self.run_bzr('branch a b --bind')
    self.assertEndsWith(err, 'New branch bound to a\n')
    b = branch.Branch.open('b')
    self.assertEndsWith(b.get_bound_location(), '/a/')