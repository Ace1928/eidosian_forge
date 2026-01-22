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
def test__split_by_prefix(self):
    self.assertSplitByPrefix({b'f': [(b'f', b'a'), (b'f', b'b')], b'g': [(b'g', b'b'), (b'g', b'a')]}, [b'f', b'g'], [(b'f', b'a'), (b'g', b'b'), (b'g', b'a'), (b'f', b'b')])
    self.assertSplitByPrefix({b'f': [(b'f', b'a'), (b'f', b'b')], b'g': [(b'g', b'b'), (b'g', b'a')]}, [b'f', b'g'], [(b'f', b'a'), (b'f', b'b'), (b'g', b'b'), (b'g', b'a')])
    self.assertSplitByPrefix({b'f': [(b'f', b'a'), (b'f', b'b')], b'g': [(b'g', b'b'), (b'g', b'a')]}, [b'f', b'g'], [(b'f', b'a'), (b'f', b'b'), (b'g', b'b'), (b'g', b'a')])
    self.assertSplitByPrefix({b'f': [(b'f', b'a'), (b'f', b'b')], b'g': [(b'g', b'b'), (b'g', b'a')], b'': [(b'a',), (b'b',)]}, [b'f', b'g', b''], [(b'f', b'a'), (b'g', b'b'), (b'a',), (b'b',), (b'g', b'a'), (b'f', b'b')])