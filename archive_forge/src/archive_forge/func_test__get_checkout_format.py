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
def test__get_checkout_format(self):
    transport = MemoryTransport()
    client = FakeClient(transport.base)
    reference_bzrdir_format = controldir.format_registry.get('default')()
    control_name = reference_bzrdir_format.network_name()
    client.add_expected_call(b'BzrDir.checkout_metadir', (b'quack/',), b'success', (control_name, b'', b''))
    transport.mkdir('quack')
    transport = transport.clone('quack')
    a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
    result = a_controldir.checkout_metadir()
    self.assertEqual(bzrdir.BzrDirMetaFormat1, type(result))
    self.assertEqual(None, result._repository_format)
    self.assertEqual(None, result._branch_format)
    self.assertFinished(client)