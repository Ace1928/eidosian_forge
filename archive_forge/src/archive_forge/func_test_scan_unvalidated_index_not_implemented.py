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
def test_scan_unvalidated_index_not_implemented(self):
    transport = MockTransport()
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertRaises(NotImplementedError, index.scan_unvalidated_index, 'dummy graph_index')
    self.assertRaises(NotImplementedError, index.get_missing_compression_parents)