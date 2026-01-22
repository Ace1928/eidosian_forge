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
def test_history_incomplete_with_fallback(self):
    """A 'history-incomplete' response causes the fallback repository to be
        queried too, if one is set.
        """
    format = remote.response_tuple_to_repo_format((b'yes', b'no', b'yes', self.get_repo_format().network_name()))
    repo, client = self.setup_fake_client_and_repository('quack')
    repo._format = format
    fallback_repo, ignored = self.setup_fake_client_and_repository('fallback')
    fallback_repo._client = client
    fallback_repo._format = format
    repo.add_fallback_repository(fallback_repo)
    client.add_expected_call(b'Repository.get_rev_id_for_revno', (b'quack/', 1, (42, b'rev-foo')), b'success', (b'history-incomplete', 2, b'rev-two'))
    client.add_expected_call(b'Repository.get_rev_id_for_revno', (b'fallback/', 1, (2, b'rev-two')), b'success', (b'ok', b'rev-one'))
    result = repo.get_rev_id_for_revno(1, (42, b'rev-foo'))
    self.assertEqual((True, b'rev-one'), result)
    self.assertFinished(client)