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
def test_some_revisions(self):
    """An empty body should be returned for an empty repository."""
    backing = self.get_transport()
    request = smart_repo.SmartServerRepositoryAllRevisionIds(backing)
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    tree.commit(rev_id=b'origineel', message='message')
    tree.commit(rev_id=b'nog-een-revisie', message='message')
    tree.unlock()
    self.assertIn(request.execute(b''), [smart_req.SuccessfulSmartServerResponse((b'ok',), b'origineel\nnog-een-revisie'), smart_req.SuccessfulSmartServerResponse((b'ok',), b'nog-een-revisie\norigineel')])