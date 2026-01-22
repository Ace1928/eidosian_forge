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
def test_remote_path_from_transport_http(self):
    """Remote paths for HTTP transports are calculated differently to other
        transports.  They are just relative to the client base, not the root
        directory of the host.
        """
    for scheme in ['http:', 'https:', 'bzr+http:', 'bzr+https:']:
        self.assertRemotePathHTTP('../xyz/', scheme + '//host/path', '../xyz/')
        self.assertRemotePathHTTP('xyz/', scheme + '//host/path', 'xyz/')