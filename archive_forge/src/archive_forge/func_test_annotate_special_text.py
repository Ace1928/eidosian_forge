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
def test_annotate_special_text(self):
    ann = self.make_annotator()
    vf = ann._vf
    rev1_key = (b'rev-1',)
    rev2_key = (b'rev-2',)
    rev3_key = (b'rev-3',)
    spec_key = (b'special:',)
    vf.add_lines(rev1_key, [], [b'initial content\n'])
    vf.add_lines(rev2_key, [rev1_key], [b'initial content\n', b'common content\n', b'content in 2\n'])
    vf.add_lines(rev3_key, [rev1_key], [b'initial content\n', b'common content\n', b'content in 3\n'])
    spec_text = b'initial content\ncommon content\ncontent in 2\ncontent in 3\n'
    ann.add_special_text(spec_key, [rev2_key, rev3_key], spec_text)
    anns, lines = ann.annotate(spec_key)
    self.assertEqual([(rev1_key,), (rev2_key, rev3_key), (rev2_key,), (rev3_key,)], anns)
    self.assertEqualDiff(spec_text, b''.join(lines))