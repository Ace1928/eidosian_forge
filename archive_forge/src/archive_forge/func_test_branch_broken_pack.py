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
def test_branch_broken_pack(self):
    """branching with a corrupted pack file."""
    self.example_branch('a')
    packs_dir = 'a/.bzr/repository/packs/'
    fname = packs_dir + os.listdir(packs_dir)[0]
    with open(fname, 'rb+') as f:
        f.seek(-5, os.SEEK_END)
        c = f.read(1)
        f.seek(-5, os.SEEK_END)
        if c == b'\xff':
            corrupt = b'\x00'
        else:
            corrupt = b'\xff'
        f.write(corrupt)
    self.run_bzr_error(['Corruption while decompressing repository file'], 'branch a b', retcode=3)