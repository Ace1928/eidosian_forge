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
def test_fast_delta(self):
    true_name = groupcompress_repo.RepositoryFormat2a().network_name()
    true_format = RemoteRepositoryFormat()
    true_format._network_name = true_name
    self.assertEqual(True, true_format.fast_deltas)
    false_name = knitpack_repo.RepositoryFormatKnitPack1().network_name()
    false_format = RemoteRepositoryFormat()
    false_format._network_name = false_name
    self.assertEqual(False, false_format.fast_deltas)