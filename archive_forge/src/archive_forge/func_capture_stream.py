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
def capture_stream(self, f, entries, on_seen, parents, require_fulltext=False):
    """Capture a stream for testing."""
    for factory in entries:
        on_seen(factory.key)
        self.assertValidStorageKind(factory.storage_kind)
        if factory.sha1 is not None:
            self.assertEqual(f.get_sha1s([factory.key])[factory.key], factory.sha1)
        self.assertEqual(parents[factory.key], factory.parents)
        self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
        if require_fulltext:
            factory.get_bytes_as('fulltext')