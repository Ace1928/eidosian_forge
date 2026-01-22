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
def test_lock_write_on_unlocked_branch(self):
    backing = self.get_transport()
    request = smart_branch.SmartServerBranchRequestLockWrite(backing)
    branch = self.make_branch('.', format='knit')
    repository = branch.repository
    response = request.execute(b'')
    branch_nonce = branch.control_files._lock.peek().nonce
    repository_nonce = repository.control_files._lock.peek().nonce
    self.assertEqual(smart_req.SmartServerResponse((b'ok', branch_nonce, repository_nonce)), response)
    new_branch = repository.controldir.open_branch()
    self.assertRaises(errors.LockContention, new_branch.lock_write)
    request = smart_branch.SmartServerBranchRequestUnlock(backing)
    response = request.execute(b'', branch_nonce, repository_nonce)