import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def make_branch_with_divergent_history(self):
    """Make a branch with divergent history in its repo.

        The branch's tip will be 'child-2', and the repo will also contain
        'child-1', which diverges from a common base revision.
        """
    self.tree.lock_write()
    self.tree.add('')
    self.tree.commit('1st commit')
    revno_1, revid_1 = self.tree.branch.last_revision_info()
    self.tree.commit('2nd commit', rev_id=b'child-1')
    self.tree.branch.set_last_revision_info(revno_1, revid_1)
    self.tree.set_parent_ids([revid_1])
    self.tree.commit('2nd commit', rev_id=b'child-2')
    self.tree.unlock()