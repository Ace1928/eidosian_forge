import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
def test_tip_change_rejected(self):
    """TipChangeRejected responses cause a TipChangeRejected exception to
        be raised.
        """
    transport = MemoryTransport()
    transport.mkdir('branch')
    transport = transport.clone('branch')
    client = FakeClient(transport.base)
    client.add_error_response(b'NotStacked')
    client.add_success_response(b'ok', b'branch token', b'repo token')
    client.add_error_response(b'TipChangeRejected', b'rejection message')
    client.add_success_response(b'ok')
    branch = self.make_remote_branch(transport, client)
    branch.lock_write()
    self.addCleanup(branch.unlock)
    client._calls = []
    err = self.assertRaises(errors.TipChangeRejected, branch.set_last_revision_info, 123, b'revid')
    self.assertEqual('rejection message', err.msg)