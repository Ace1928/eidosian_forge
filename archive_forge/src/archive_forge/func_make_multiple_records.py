import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def make_multiple_records(self):
    """Create the content for multiple records."""
    sha1sum = osutils.sha_string(b'foo\nbar\n')
    total_txt = []
    gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
    record_1 = (0, len(gz_txt), sha1sum)
    total_txt.append(gz_txt)
    sha1sum = osutils.sha_string(b'baz\n')
    gz_txt = self.create_gz_content(b'version rev-id-2 1 %s\nbaz\nend rev-id-2\n' % (sha1sum,))
    record_2 = (record_1[1], len(gz_txt), sha1sum)
    total_txt.append(gz_txt)
    return (total_txt, record_1, record_2)