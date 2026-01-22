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
def test_get_parent_map_gets_ghosts_from_result(self):
    self.setup_smart_server_with_call_log()
    tree = self.make_branch_and_memory_tree('foo')
    with tree.lock_write():
        builder = treebuilder.TreeBuilder()
        builder.start_tree(tree)
        builder.build([])
        builder.finish_tree()
        tree.set_parent_ids([b'non-existant'], allow_leftmost_as_ghost=True)
        rev_id = tree.commit('')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    repo = tree.branch.repository
    self.assertIsInstance(repo, RemoteRepository)
    repo.get_parent_map([rev_id])
    self.reset_smart_call_log()
    self.assertEqual({}, repo.get_parent_map([b'non-existant']))
    self.assertLength(0, self.hpss_calls)