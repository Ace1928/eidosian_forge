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
def test_trivial_include_missing(self):
    backing = self.get_transport()
    request = smart_repo.SmartServerRepositoryGetParentMap(backing)
    self.make_branch_and_memory_tree('.')
    self.assertEqual(None, request.execute(b'', b'missing-id', b'include-missing:'))
    self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',), bz2.compress(b'missing:missing-id')), request.do_body(b'\n\n0\n'))