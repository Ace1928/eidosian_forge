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
def test_get_position(self):
    index = self.two_graph_index()
    self.assertEqual((index._graph_index._indices[0], 0, 100), index.get_position((b'tip',)))
    self.assertEqual((index._graph_index._indices[1], 100, 78), index.get_position((b'parent',)))