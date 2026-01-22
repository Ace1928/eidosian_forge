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
def test__remember_remote_is_before(self):
    """Calling _remember_remote_is_before ratchets down the known remote
        version.
        """
    client_medium = medium.SmartClientMedium('dummy base')
    client_medium._remember_remote_is_before((1, 6))
    self.assertTrue(client_medium._is_remote_before((1, 6)))
    self.assertFalse(client_medium._is_remote_before((1, 5)))
    client_medium._remember_remote_is_before((1, 5))
    self.assertTrue(client_medium._is_remote_before((1, 5)))
    self.assertNotContainsRe(self.get_log(), '_remember_remote_is_before')
    client_medium._remember_remote_is_before((1, 9))
    self.assertContainsRe(self.get_log(), '_remember_remote_is_before')
    self.assertTrue(client_medium._is_remote_before((1, 5)))