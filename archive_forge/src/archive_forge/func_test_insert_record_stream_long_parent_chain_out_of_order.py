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
def test_insert_record_stream_long_parent_chain_out_of_order(self):
    """An out of order stream can either error or work."""
    if not self.graph:
        raise TestNotApplicable('ancestry info only relevant with graph.')
    source = self.get_versionedfiles('source')
    parents = ()
    keys = []
    content = [b'same same %d\n' % n for n in range(500)]
    letters = b'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(letters)):
        letter = letters[i:i + 1]
        key = (b'key-' + letter,)
        if self.key_length == 2:
            key = (b'prefix',) + key
        content.append(b'content for ' + letter + b'\n')
        source.add_lines(key, parents, content)
        keys.append(key)
        parents = (key,)
    streams = []
    for key in reversed(keys):
        streams.append(source.get_record_stream([key], 'unordered', False))
    deltas = itertools.chain.from_iterable(streams[:-1])
    files = self.get_versionedfiles()
    try:
        files.insert_record_stream(deltas)
    except RevisionNotPresent:
        files.check()
    else:
        missing = set(files.get_missing_compression_parent_keys())
        missing.discard(keys[0])
        self.assertEqual(set(), missing)