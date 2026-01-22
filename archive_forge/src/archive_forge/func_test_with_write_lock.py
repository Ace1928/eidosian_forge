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
def test_with_write_lock(self):
    backing = self.get_transport()
    repo = self.make_repository('.')
    self.addCleanup(repo.lock_write().unlock)
    if repo.get_physical_lock_status():
        expected = b'yes'
    else:
        expected = b'no'
    request_class = smart_repo.SmartServerRepositoryGetPhysicalLockStatus
    request = request_class(backing)
    self.assertEqual(smart_req.SuccessfulSmartServerResponse((expected,)), request.execute(b''))