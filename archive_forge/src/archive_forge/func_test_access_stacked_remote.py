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
def test_access_stacked_remote(self):
    base_builder = self.make_branch_builder('base', format='1.9')
    base_builder.start_series()
    base_revid = base_builder.build_snapshot(None, [('add', ('', None, 'directory', None))], 'message', revision_id=b'rev-id')
    base_builder.finish_series()
    stacked_branch = self.make_branch('stacked', format='1.9')
    stacked_branch.set_stacked_on_url('../base')
    smart_server = test_server.SmartTCPServer_for_testing()
    self.start_server(smart_server)
    remote_bzrdir = BzrDir.open(smart_server.get_url() + '/stacked')
    remote_branch = remote_bzrdir.open_branch()
    remote_repo = remote_branch.repository
    remote_repo.lock_read()
    try:
        self.assertLength(1, remote_repo._fallback_repositories)
        self.assertIsInstance(remote_repo._fallback_repositories[0], RemoteRepository)
        self.assertTrue(remote_repo.has_revisions([base_revid]))
        self.assertTrue(remote_repo.has_revision(base_revid))
        self.assertEqual(remote_repo.get_revision(base_revid).message, 'message')
    finally:
        remote_repo.unlock()