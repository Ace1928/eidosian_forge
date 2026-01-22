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
def test_add_lines_return_value(self):
    vf = self.get_file()
    empty_text = (b'a', [])
    sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
    sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
    for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
        result = vf.add_lines(version, [], lines)
        self.assertEqual(3, len(result))
        self.assertEqual((osutils.sha_strings(lines), sum(map(len, lines))), result[0:2])
    lines = sample_text_nl[1]
    self.assertEqual((osutils.sha_strings(lines), sum(map(len, lines))), vf.add_lines(b'd', [b'b', b'c'], lines)[0:2])