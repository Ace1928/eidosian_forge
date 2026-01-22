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
def test_branch_last_revision_info_is_updated(self):
    """A branch's tip can be set to a revision that is present in its
        repository.
        """
    self.make_tree_with_two_commits()
    rev_id_utf8 = 'È'.encode()
    self.tree.branch.set_last_revision_info(0, b'null:')
    self.assertEqual((0, b'null:'), self.tree.branch.last_revision_info())
    self.assertRequestSucceeds(rev_id_utf8, 1)
    self.assertEqual((1, rev_id_utf8), self.tree.branch.last_revision_info())