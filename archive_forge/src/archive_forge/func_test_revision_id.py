from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_revision_id(self):
    r = Revision(b'myrevid')
    r.message = 'MyCommitMessage'
    r.parent_ids = []
    r.committer = 'Jelmer Vernooij <jelmer@apache.org>'
    r.timestamp = 453543543
    r.timezone = 0
    r.properties = {}
    self.assertRoundtripRevision(r)