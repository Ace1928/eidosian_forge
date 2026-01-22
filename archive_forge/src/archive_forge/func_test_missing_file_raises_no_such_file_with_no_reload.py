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
def test_missing_file_raises_no_such_file_with_no_reload(self):
    memos = self.make_pack_file()
    transport = self.get_transport()
    access = pack_repo._DirectPackAccess({'foo': (transport, 'different-packname')})
    e = self.assertListRaises(_mod_transport.NoSuchFile, access.get_raw_records, memos)