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
def test_branch_stacked_from_non_stacked_format(self):
    """The origin format doesn't support stacking"""
    trunk = self.make_branch('trunk', format='pack-0.92')
    out, err = self.run_bzr(['branch', '--stacked', 'trunk', 'shallow'])
    self.assertEqualDiff('Source repository format does not support stacking, using format:\n  Packs 5 (adds stacking support, requires bzr 1.6)\nSource branch format does not support stacking, using format:\n  Branch format 7\nDoing on-the-fly conversion from RepositoryFormatKnitPack1() to RepositoryFormatKnitPack5().\nThis may take some time. Upgrade the repositories to the same format for better performance.\nCreated new stacked branch referring to %s.\n' % (trunk.base,), err)