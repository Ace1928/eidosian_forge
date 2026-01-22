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
def test_not_empty(self):
    """For a non-empty branch, the result is ('ok', 'revno', 'revid')."""
    backing = self.get_transport()
    request = smart_branch.SmartServerBranchRequestLastRevisionInfo(backing)
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    rev_id_utf8 = 'È'.encode()
    tree.commit('1st commit')
    tree.commit('2nd commit', rev_id=rev_id_utf8)
    tree.unlock()
    self.assertEqual(smart_req.SmartServerResponse((b'ok', b'2', rev_id_utf8)), request.execute(b''))