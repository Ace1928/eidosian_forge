from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_escape_control_l(self):
    self.assertEqual(b'bla_c', escape_file_id(b'bla\x0c'))