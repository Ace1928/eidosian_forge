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
def test_backwards_compat_hpss_v2(self):
    client, transport = self.make_fake_client_and_transport()
    orig_check_call = client._check_call

    def check_call(method, args):
        client._medium._protocol_version = 2
        client._medium._remember_remote_is_before((1, 6))
        client._check_call = orig_check_call
        client._check_call(method, args)
    client._check_call = check_call
    client.add_expected_call(b'BzrDir.open_2.1', (b'quack/',), b'unknown', (b'BzrDir.open_2.1',))
    client.add_expected_call(b'BzrDir.open', (b'quack/',), b'success', (b'yes',))
    bd = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client, _force_probe=True)
    self.assertIsInstance(bd, RemoteBzrDir)
    self.assertFinished(client)