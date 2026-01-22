import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def test_get_record_stream(self):
    self._lines[b'A'] = [b'FOO', b'BAR']
    it = self.texts.get_record_stream([(b'A',)], 'unordered', True)
    record = next(it)
    self.assertEqual('chunked', record.storage_kind)
    self.assertEqual(b'FOOBAR', record.get_bytes_as('fulltext'))
    self.assertEqual([b'FOO', b'BAR'], record.get_bytes_as('chunked'))