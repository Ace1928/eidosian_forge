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
def test_makes_repository(self):
    """When there is a bzrdir present, the call succeeds."""
    backing = self.get_transport()
    self.make_controldir('.')
    request_class = smart_dir.SmartServerRequestCreateRepository
    request = request_class(backing)
    reference_bzrdir_format = controldir.format_registry.get('pack-0.92')()
    reference_format = reference_bzrdir_format.repository_format
    network_name = reference_format.network_name()
    expected = smart_req.SuccessfulSmartServerResponse((b'ok', b'no', b'no', b'no', network_name))
    self.assertEqual(expected, request.execute(b'', network_name, b'True'))