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
def test_backwards_compatible(self):
    self.make_branch('base', format='1.6')
    stacked_branch = self.make_branch('stacked', format='1.6')
    stacked_branch.set_stacked_on_url('../base')
    client = FakeClient(self.get_url())
    branch_network_name = self.get_branch_format().network_name()
    client.add_expected_call(b'BzrDir.open_branchV3', (b'stacked/',), b'success', (b'branch', branch_network_name))
    client.add_expected_call(b'BzrDir.find_repositoryV3', (b'stacked/',), b'success', (b'ok', b'', b'no', b'no', b'yes', stacked_branch.repository._format.network_name()))
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'stacked/',), b'unknown', (b'Branch.get_stacked_on_url',))
    client.add_expected_call(b'Branch.get_stacked_on_url', (b'stacked/',), b'unknown', (b'Branch.get_stacked_on_url',))
    bzrdir = RemoteBzrDir(self.get_transport('stacked'), RemoteBzrDirFormat(), _client=client)
    branch = bzrdir.open_branch()
    result = branch.get_stacked_on_url()
    self.assertEqual('../base', result)
    self.assertFinished(client)
    self.assertEqual(1, len(branch.repository._fallback_repositories))
    self.assertEqual(1, len(branch.repository._real_repository._fallback_repositories))