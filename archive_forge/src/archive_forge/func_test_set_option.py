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
def test_set_option(self):
    client = FakeClient()
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
    client.add_expected_call(b'Branch.lock_write', (b'memory:///', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
    client.add_expected_call(b'Branch.set_config_option', (b'memory:///', b'branch token', b'repo token', b'foo', b'bar', b''), b'success', ())
    client.add_expected_call(b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
    transport = MemoryTransport()
    branch = self.make_remote_branch(transport, client)
    branch.lock_write()
    config = branch._get_config()
    config.set_option('foo', 'bar')
    branch.unlock()
    self.assertFinished(client)