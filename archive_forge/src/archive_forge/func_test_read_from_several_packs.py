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
def test_read_from_several_packs(self):
    access, writer = self._get_access()
    memos = []
    memos.extend(access.add_raw_records([(b'key', 10)], [b'1234567890']))
    writer.end()
    access, writer = self._get_access('pack2', 'FOOBAR')
    memos.extend(access.add_raw_records([(b'key', 5)], [b'12345']))
    writer.end()
    access, writer = self._get_access('pack3', 'BAZ')
    memos.extend(access.add_raw_records([(b'key', 5)], [b'alpha']))
    writer.end()
    transport = self.get_transport()
    access = pack_repo._DirectPackAccess({'FOO': (transport, 'packfile'), 'FOOBAR': (transport, 'pack2'), 'BAZ': (transport, 'pack3')})
    self.assertEqual([b'1234567890', b'12345', b'alpha'], list(access.get_raw_records(memos)))
    self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[0:1])))
    self.assertEqual([b'12345'], list(access.get_raw_records(memos[1:2])))
    self.assertEqual([b'alpha'], list(access.get_raw_records(memos[2:3])))
    self.assertEqual([b'1234567890', b'alpha'], list(access.get_raw_records(memos[0:1] + memos[2:3])))