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
def test_stacked_branch(self):
    """Opening a stacked branch does not open the stacked-on branch."""
    trunk = self.make_branch('trunk')
    feature = self.make_branch('feature')
    feature.set_stacked_on_url(trunk.base)
    opened_branches = []
    _mod_branch.Branch.hooks.install_named_hook('open', opened_branches.append, None)
    backing = self.get_transport()
    request = smart_dir.SmartServerRequestOpenBranchV3(backing)
    request.setup_jail()
    try:
        response = request.execute(b'feature')
    finally:
        request.teardown_jail()
    expected_format = feature._format.network_name()
    self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'branch', expected_format)), response)
    self.assertLength(1, opened_branches)