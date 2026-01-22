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
def test_unrecognised_server_error(self):
    """If the error code from the server is not recognised, the original
        ErrorFromSmartServer is propagated unmodified.
        """
    error_tuple = (b'An unknown error tuple',)
    server_error = errors.ErrorFromSmartServer(error_tuple)
    translated_error = self.translateErrorFromSmartServer(server_error)
    expected_error = UnknownErrorFromSmartServer(server_error)
    self.assertEqual(expected_error, translated_error)