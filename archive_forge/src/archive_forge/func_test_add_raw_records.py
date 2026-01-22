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
def test_add_raw_records(self):
    """add_raw_records adds records retrievable later."""
    access = self.get_access()
    memos = access.add_raw_records([(b'key', 10)], [b'1234567890'])
    self.assertEqual([b'1234567890'], list(access.get_raw_records(memos)))