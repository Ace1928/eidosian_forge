from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_simple_commit(self):
    r = Revision(self.mapping.revision_id_foreign_to_bzr(b'edf99e6c56495c620f20d5dacff9859ff7119261'))
    r.message = 'MyCommitMessage'
    r.parent_ids = []
    r.committer = 'Jelmer Vernooij <jelmer@apache.org>'
    r.timestamp = 453543543
    r.timezone = 0
    r.properties = {}
    self.assertRoundtripRevision(r)