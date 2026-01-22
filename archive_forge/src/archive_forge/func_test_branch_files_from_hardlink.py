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
def test_branch_files_from_hardlink(self):
    self.requireFeature(HardlinkFeature(self.test_dir))
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file1'])
    source.add('file1')
    source.commit('added file')
    source.controldir.sprout('second')
    out, err = self.run_bzr('branch source target --files-from second --hardlink')
    source_stat = os.stat('source/file1')
    second_stat = os.stat('second/file1')
    target_stat = os.stat('target/file1')
    self.assertNotEqual(source_stat, target_stat)
    self.assertEqual(second_stat, target_stat)