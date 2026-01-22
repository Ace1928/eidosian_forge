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
def test_insert_record_stream_out_of_order(self):
    """An out of order stream can either error or work."""
    files = self.get_versionedfiles()
    source = self.get_versionedfiles('source')
    self.get_diamond_files(source)
    if self.key_length == 1:
        origin_keys = [(b'origin',)]
        end_keys = [(b'merged',), (b'left',)]
        start_keys = [(b'right',), (b'base',)]
    else:
        origin_keys = [(b'FileA', b'origin'), (b'FileB', b'origin')]
        end_keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileB', b'merged'), (b'FileB', b'left')]
        start_keys = [(b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'right'), (b'FileB', b'base')]
    origin_entries = source.get_record_stream(origin_keys, 'unordered', False)
    end_entries = source.get_record_stream(end_keys, 'topological', False)
    start_entries = source.get_record_stream(start_keys, 'topological', False)
    entries = itertools.chain(origin_entries, end_entries, start_entries)
    try:
        files.insert_record_stream(entries)
    except RevisionNotPresent:
        files.check()
    else:
        self.assertIdenticalVersionedFile(source, files)