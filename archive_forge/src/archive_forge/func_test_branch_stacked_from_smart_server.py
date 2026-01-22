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
def test_branch_stacked_from_smart_server(self):
    self.transport_server = test_server.SmartTCPServer_for_testing
    trunk = self.make_branch('mainline', format='1.9')
    out, err = self.run_bzr(['branch', '--stacked', self.get_url('mainline'), 'shallow'])