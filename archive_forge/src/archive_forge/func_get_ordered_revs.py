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
def get_ordered_revs(self, format, order, branch_factory=None):
    """Get a list of the revisions in a stream to format format.

        :param format: The format of the target.
        :param order: the order that target should have requested.
        :param branch_factory: A callable to create a trunk and stacked branch
            to fetch from. If none, self.prepare_stacked_remote_branch is used.
        :result: The revision ids in the stream, in the order seen,
            the topological order of revisions in the source.
        """
    unordered_format = controldir.format_registry.get(format)()
    target_repository_format = unordered_format.repository_format
    self.assertEqual(order, target_repository_format._fetch_order)
    if branch_factory is None:
        branch_factory = self.prepare_stacked_remote_branch
    _, stacked = branch_factory()
    source = stacked.repository._get_source(target_repository_format)
    tip = stacked.last_revision()
    stacked.repository._ensure_real()
    graph = stacked.repository.get_graph()
    revs = [r for r, ps in graph.iter_ancestry([tip]) if r != NULL_REVISION]
    revs.reverse()
    search = vf_search.PendingAncestryResult([tip], stacked.repository)
    self.reset_smart_call_log()
    stream = source.get_stream(search)
    return (self.fetch_stream_to_rev_order(stream), revs)