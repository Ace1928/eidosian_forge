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
def test_check_write_group_invalid(self):
    backing = self.get_transport()
    repo = self.make_repository('.')
    lock_token = repo.lock_write().repository_token
    self.addCleanup(repo.unlock)
    request_class = smart_repo.SmartServerRepositoryCheckWriteGroup
    request = request_class(backing)
    self.assertEqual(smart_req.FailedSmartServerResponse((b'UnresumableWriteGroup', [b'random'], b'Malformed write group token')), request.execute(b'', lock_token, [b'random']))