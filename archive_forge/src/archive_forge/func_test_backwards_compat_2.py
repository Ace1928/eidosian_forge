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
def test_backwards_compat_2(self):
    reference_format = self.get_repo_format()
    network_name = reference_format.network_name()
    server_url = 'bzr://example.com/'
    self.permit_url(server_url)
    client = FakeClient(server_url)
    client.add_unknown_method_response(b'BzrDir.find_repositoryV3')
    client.add_success_response(b'ok', b'', b'no', b'no', b'no')
    client.add_success_response_with_body(b'Bazaar-NG meta directory, format 1\n', b'ok')
    client.add_success_response(b'stat', b'0', b'65535')
    client.add_success_response_with_body(reference_format.get_format_string(), b'ok')
    client.add_success_response(b'stat', b'0', b'65535')
    remote_transport = RemoteTransport(server_url + 'quack/', medium=False, _client=client)
    bzrdir = RemoteBzrDir(remote_transport, RemoteBzrDirFormat(), _client=client)
    repo = bzrdir.open_repository()
    self.assertEqual([('call', b'BzrDir.find_repositoryV3', (b'quack/',)), ('call', b'BzrDir.find_repositoryV2', (b'quack/',)), ('call_expecting_body', b'get', (b'/quack/.bzr/branch-format',)), ('call', b'stat', (b'/quack/.bzr',)), ('call_expecting_body', b'get', (b'/quack/.bzr/repository/format',)), ('call', b'stat', (b'/quack/.bzr/repository',))], client._calls)
    self.assertEqual(network_name, repo._format.network_name())