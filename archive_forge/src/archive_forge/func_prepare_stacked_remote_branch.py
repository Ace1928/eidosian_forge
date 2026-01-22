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
def prepare_stacked_remote_branch(self):
    """Get stacked_upon and stacked branches with content in each."""
    self.setup_smart_server_with_call_log()
    tree1 = self.make_branch_and_tree('tree1', format='1.9')
    tree1.commit('rev1', rev_id=b'rev1')
    tree2 = tree1.branch.controldir.sprout('tree2', stacked=True).open_workingtree()
    local_tree = tree2.branch.create_checkout('local')
    local_tree.commit('local changes make me feel good.')
    branch2 = Branch.open(self.get_url('tree2'))
    branch2.lock_read()
    self.addCleanup(branch2.unlock)
    return (tree1.branch, branch2)