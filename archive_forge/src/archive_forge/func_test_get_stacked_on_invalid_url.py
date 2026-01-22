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
def test_get_stacked_on_invalid_url(self):
    stacked_branch = self.make_branch('stacked', format='1.9')
    self.make_branch('base', format='1.9')
    vfs_url = self.get_vfs_only_url('base')
    stacked_branch.set_stacked_on_url(vfs_url)
    transport = stacked_branch.controldir.root_transport
    client = FakeClient(transport.base)
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'stacked/',), b'success', (b'ok', vfs_url.encode('utf-8')))
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'stacked/',), b'success', (b'ok', vfs_url.encode('utf-8')))
    bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
    repo_fmt = remote.RemoteRepositoryFormat()
    repo_fmt._custom_format = stacked_branch.repository._format
    branch = RemoteBranch(bzrdir, RemoteRepository(bzrdir, repo_fmt), _client=client)
    result = branch.get_stacked_on_url()
    self.assertEqual(vfs_url, result)