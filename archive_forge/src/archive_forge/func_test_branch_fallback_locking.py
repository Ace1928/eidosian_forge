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
def test_branch_fallback_locking(self):
    """RemoteBranch.get_rev_id takes a read lock, and tries to call the
        get_rev_id_for_revno verb.  If the verb is unknown the VFS fallback
        will be invoked, which will fail if the repo is unlocked.
        """
    self.setup_smart_server_with_call_log()
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    rev1 = tree.commit('First')
    tree.commit('Second')
    tree.unlock()
    branch = tree.branch
    self.assertFalse(branch.is_locked())
    self.reset_smart_call_log()
    verb = b'Repository.get_rev_id_for_revno'
    self.disable_verb(verb)
    self.assertEqual(rev1, branch.get_rev_id(1))
    self.assertLength(1, [call for call in self.hpss_calls if call.call.method == verb])