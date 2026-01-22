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
def test__str__(self):
    """SmartServerResponses can be stringified."""
    self.assertIn(str(smart_req.SuccessfulSmartServerResponse((b'args',), b'body')), ("<SuccessfulSmartServerResponse args=(b'args',) body=b'body'>", "<SuccessfulSmartServerResponse args=('args',) body='body'>"))
    self.assertIn(str(smart_req.FailedSmartServerResponse((b'args',), b'body')), ("<FailedSmartServerResponse args=(b'args',) body=b'body'>", "<FailedSmartServerResponse args=('args',) body='body'>"))