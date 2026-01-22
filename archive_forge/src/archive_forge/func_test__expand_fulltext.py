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
def test__expand_fulltext(self):
    ann = self.make_annotator()
    rev_key = (b'rev-id',)
    ann._num_compression_children[rev_key] = 1
    res = ann._expand_record(rev_key, ((b'parent-id',),), None, [b'line1\n', b'line2\n'], ('fulltext', True))
    self.assertEqual([b'line1\n', b'line2'], res)
    content_obj = ann._content_objects[rev_key]
    self.assertEqual([b'line1\n', b'line2\n'], content_obj._lines)
    self.assertEqual(res, content_obj.text())
    self.assertEqual(res, ann._text_cache[rev_key])