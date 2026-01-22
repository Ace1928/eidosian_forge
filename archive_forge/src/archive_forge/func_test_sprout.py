from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_sprout(self):
    """Test we can clone dummies and that the format is not preserved."""
    self.make_branch_and_tree('d', format=DummyForeignVcsDirFormat())
    dir = controldir.ControlDir.open('d')
    newdir = dir.sprout('e')
    self.assertNotEqual(b'A Dummy VCS Dir', newdir._format.get_format_string())