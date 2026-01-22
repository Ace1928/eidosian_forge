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
def test_from_name(self):
    """Branch from a colocated branch into a regular branch."""
    os.mkdir('b')
    tree = self.example_branch('b/a', format='development-colo')
    tree.controldir.create_branch(name='somecolo')
    out, err = self.run_bzr('branch -b somecolo %s' % local_path_to_url('b/a'))
    self.assertEqual('', out)
    self.assertEqual('Branched 0 revisions.\n', err)
    self.assertPathExists('a')