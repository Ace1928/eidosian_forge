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
def test_filter_absent_records(self):
    """Requested missing records can be filter trivially."""
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    keys, _ = self.get_keys_and_sort_order()
    parent_map = files.get_parent_map(keys)
    present_keys = list(keys)
    if self.key_length == 1:
        keys.insert(2, (b'extra',))
    else:
        keys.insert(2, (b'extra', b'extra'))
    entries = files.get_record_stream(keys, 'unordered', False)
    seen = set()
    self.capture_stream(files, versionedfile.filter_absent(entries), seen.add, parent_map)
    self.assertEqual(set(present_keys), seen)