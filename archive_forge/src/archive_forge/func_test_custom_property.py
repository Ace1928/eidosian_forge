from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_custom_property(self):
    r = Revision(b'myrevid')
    r.message = 'MyCommitMessage'
    r.parent_ids = []
    r.properties = {'fool': 'bar'}
    r.committer = 'Jelmer Vernooij <jelmer@apache.org>'
    r.timestamp = 453543543
    r.timezone = 0
    self.assertRoundtripRevision(r)