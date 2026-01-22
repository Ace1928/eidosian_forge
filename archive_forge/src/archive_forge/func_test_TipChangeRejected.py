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
def test_TipChangeRejected(self):
    """If a pre_change_branch_tip hook raises TipChangeRejected, the verb
        returns TipChangeRejected.
        """
    rejection_message = 'rejection message‽'

    def hook_that_rejects(params):
        raise errors.TipChangeRejected(rejection_message)
    _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_rejects, None)
    self.assertEqual(smart_req.FailedSmartServerResponse((b'TipChangeRejected', rejection_message.encode('utf-8'))), self.set_last_revision(b'null:', 0))