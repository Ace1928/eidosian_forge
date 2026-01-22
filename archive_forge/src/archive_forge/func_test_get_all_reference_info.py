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
def test_get_all_reference_info(self):
    transport = MemoryTransport()
    client = FakeClient(transport.base)
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
    client.add_expected_call(b'Branch.get_all_reference_info', (b'quack/',), b'success', (b'ok',), bencode.bencode([(b'file-id', b'https://www.example.com/', b'')]))
    transport.mkdir('quack')
    transport = transport.clone('quack')
    branch = self.make_remote_branch(transport, client)
    result = branch._get_all_reference_info()
    self.assertFinished(client)
    self.assertEqual({b'file-id': ('https://www.example.com/', None)}, result)