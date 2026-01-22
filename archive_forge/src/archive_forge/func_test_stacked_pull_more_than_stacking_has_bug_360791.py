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
def test_stacked_pull_more_than_stacking_has_bug_360791(self):
    self.setup_smart_server_with_call_log()
    trunk = self.make_branch_and_tree('trunk', format='1.9-rich-root')
    trunk.commit('start')
    stacked_branch = trunk.branch.create_clone_on_transport(self.get_transport('stacked'), stacked_on=trunk.branch.base)
    local = self.make_branch('local', format='1.9-rich-root')
    local.repository.fetch(stacked_branch.repository, stacked_branch.last_revision())