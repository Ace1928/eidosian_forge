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
def test_get_parent_map_fallback_parentless_node(self):
    """get_parent_map falls back to get_revision_graph on old servers.  The
        results from get_revision_graph are tweaked to match the get_parent_map
        API.

        Specifically, a {key: ()} result from get_revision_graph means "no
        parents" for that key, which in get_parent_map results should be
        represented as {key: ('null:',)}.

        This is the test for https://bugs.launchpad.net/bzr/+bug/214894
        """
    rev_id = b'revision-id'
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_success_response_with_body(rev_id, b'ok')
    client._medium._remember_remote_is_before((1, 2))
    parents = repo.get_parent_map([rev_id])
    self.assertEqual([('call_expecting_body', b'Repository.get_revision_graph', (b'quack/', b''))], client._calls)
    self.assertEqual({rev_id: (b'null:',)}, parents)