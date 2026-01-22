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
def test_get_record_stream_unordered_fulltexts(self):
    basis, test = self.get_basis_and_test_knit()
    key = (b'foo',)
    key_basis = (b'bar',)
    key_missing = (b'missing',)
    test.add_lines(key, (), [b'foo\n'])
    records = list(test.get_record_stream([key], 'unordered', True))
    self.assertEqual(1, len(records))
    self.assertEqual([], basis.calls)
    basis.add_lines(key_basis, (), [b'foo\n', b'bar\n'])
    basis.calls = []
    records = list(test.get_record_stream([key_basis, key_missing], 'unordered', True))
    self.assertEqual(2, len(records))
    calls = list(basis.calls)
    for record in records:
        self.assertSubset([record.key], (key_basis, key_missing))
        if record.key == key_missing:
            self.assertIsInstance(record, AbsentContentFactory)
        else:
            reference = list(basis.get_record_stream([key_basis], 'unordered', True))[0]
            self.assertEqual(reference.key, record.key)
            self.assertEqual(reference.sha1, record.sha1)
            self.assertEqual(reference.storage_kind, record.storage_kind)
            self.assertEqual(reference.get_bytes_as(reference.storage_kind), record.get_bytes_as(record.storage_kind))
            self.assertEqual(reference.get_bytes_as('fulltext'), record.get_bytes_as('fulltext'))
    self.assertEqual([('get_parent_map', {key_basis, key_missing}), ('get_record_stream', [key_basis], 'unordered', True)], calls)