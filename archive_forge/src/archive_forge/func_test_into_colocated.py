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
def test_into_colocated(self):
    """Branch from a branch into a colocated branch."""
    self.example_branch('a')
    out, err = self.run_bzr('init --format=development-colo file:b,branch=orig')
    self.assertEqual('Created a standalone tree (format: development-colo)\n', out)
    self.assertEqual('', err)
    out, err = self.run_bzr('branch a file:b,branch=thiswasa')
    self.assertEqual('', out)
    self.assertEqual('Branched 2 revisions.\n', err)
    out, err = self.run_bzr('branches b')
    self.assertEqual('  orig\n  thiswasa\n', out)
    self.assertEqual('', err)
    out, err = self.run_bzr('branch a file:b,branch=orig', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: Already a branch: "file:b,branch=orig".\n', err)