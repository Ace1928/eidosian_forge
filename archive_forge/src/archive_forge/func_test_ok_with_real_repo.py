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
def test_ok_with_real_repo(self):
    """When the server returns 'ok' and there is a _real_repository, then
        the _real_repository's reload_pack_name's method will be called.
        """
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_expected_call(b'PackRepository.autopack', (b'quack/',), b'success', (b'ok',))
    repo._real_repository = _StubRealPackRepository(client._calls)
    repo.autopack()
    self.assertEqual([('call', b'PackRepository.autopack', (b'quack/',)), ('pack collection reload_pack_names',)], client._calls)