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
def test_current_server(self):
    reference_format = self.get_repo_format()
    network_name = reference_format.network_name()
    transport = MemoryTransport()
    transport.mkdir('quack')
    transport = transport.clone('quack')
    client = FakeClient(transport.base)
    client.add_success_response(b'ok', b'', b'no', b'no', b'no', network_name)
    bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
    repo = bzrdir.open_repository()
    self.assertEqual([('call', b'BzrDir.find_repositoryV3', (b'quack/',))], client._calls)
    self.assertEqual(network_name, repo._format.network_name())