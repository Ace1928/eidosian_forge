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
def test_insert_record_stream_delta_missing_basis_can_be_added_later(self):
    """Insertion where a needed basis is not included notifies the caller
        of the missing basis.  That basis can be added in a second
        insert_record_stream call that does not need to repeat records present
        in the previous stream.  The record(s) that required that basis are
        fully inserted once their basis is no longer missing.
        """
    if not self.support_partial_insertion:
        raise TestNotApplicable('versioned file scenario does not support partial insertion')
    source = self.get_knit_delta_source()
    entries = source.get_record_stream([self.get_simple_key(b'origin'), self.get_simple_key(b'merged')], 'unordered', False)
    files = self.get_versionedfiles()
    files.insert_record_stream(entries)
    missing_bases = files.get_missing_compression_parent_keys()
    self.assertEqual({self.get_simple_key(b'left')}, set(missing_bases))
    merged_key = self.get_simple_key(b'merged')
    self.assertEqual([merged_key], list(files.get_parent_map([merged_key]).keys()))
    missing_entries = source.get_record_stream(missing_bases, 'unordered', True)
    files.insert_record_stream(missing_entries)
    self.assertEqual([], list(files.get_missing_compression_parent_keys()))
    self.assertEqual([merged_key], list(files.get_parent_map([merged_key]).keys()))
    files.check()