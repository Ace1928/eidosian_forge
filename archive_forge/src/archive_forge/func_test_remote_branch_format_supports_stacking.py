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
def test_remote_branch_format_supports_stacking(self):
    t = self.transport
    self.make_branch('unstackable', format='pack-0.92')
    b = BzrDir.open_from_transport(t.clone('unstackable')).open_branch()
    self.assertFalse(b._format.supports_stacking())
    self.make_branch('stackable', format='1.9')
    b = BzrDir.open_from_transport(t.clone('stackable')).open_branch()
    self.assertTrue(b._format.supports_stacking())