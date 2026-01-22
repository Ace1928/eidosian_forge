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
def test_ancestry(self):
    f = self.get_file()
    self.assertEqual(set(), f.get_ancestry([]))
    f.add_lines(b'r0', [], [b'a\n', b'b\n'])
    f.add_lines(b'r1', [b'r0'], [b'b\n', b'c\n'])
    f.add_lines(b'r2', [b'r0'], [b'b\n', b'c\n'])
    f.add_lines(b'r3', [b'r2'], [b'b\n', b'c\n'])
    f.add_lines(b'rM', [b'r1', b'r2'], [b'b\n', b'c\n'])
    self.assertEqual(set(), f.get_ancestry([]))
    versions = f.get_ancestry([b'rM'])
    self.assertRaises(RevisionNotPresent, f.get_ancestry, [b'rM', b'rX'])
    self.assertEqual(set(f.get_ancestry(b'rM')), set(f.get_ancestry(b'rM')))