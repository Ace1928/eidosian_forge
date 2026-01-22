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
def test_set_nonempty(self):
    transport = MemoryTransport()
    transport.mkdir('branch')
    transport = transport.clone('branch')
    client = FakeClient(transport.base)
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'branch/',), b'error', (b'NotStacked',))
    client.add_expected_call(b'Branch.lock_write', (b'branch/', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
    client.add_expected_call(b'Branch.last_revision_info', (b'branch/',), b'success', (b'ok', b'0', b'null:'))
    lines = [b'rev-id2']
    encoded_body = bz2.compress(b'\n'.join(lines))
    client.add_success_response_with_body(encoded_body, b'ok')
    client.add_expected_call(b'Branch.set_last_revision', (b'branch/', b'branch token', b'repo token', b'rev-id2'), b'success', (b'ok',))
    client.add_expected_call(b'Branch.unlock', (b'branch/', b'branch token', b'repo token'), b'success', (b'ok',))
    branch = self.make_remote_branch(transport, client)
    branch.lock_write()
    result = branch._set_last_revision(b'rev-id2')
    branch.unlock()
    self.assertEqual(None, result)
    self.assertFinished(client)