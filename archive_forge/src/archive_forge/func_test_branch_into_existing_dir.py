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
def test_branch_into_existing_dir(self):
    self.example_branch('a')
    self.build_tree_contents([('b/',)])
    self.build_tree_contents([('b/hello', b'bar')])
    self.build_tree_contents([('b/goodbye', b'baz')])
    out, err = self.run_bzr('branch a b', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: Target directory "b" already exists.\n', err)
    self.run_bzr('branch a b --use-existing-dir')
    self.assertPathExists('b/hello.moved')
    self.assertPathDoesNotExist('b/godbye.moved')
    out, err = self.run_bzr('branch a b --use-existing-dir', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: Already a branch: "b".\n', err)