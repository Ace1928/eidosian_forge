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
def test_add_lines_with_matching_blocks_noeol_last_line(self):
    """Add a text with an unchanged last line with no eol should work."""
    from breezy import multiparent
    sha1 = '6a1d115ec7b60afb664dc14890b5af5ce3c827a4'
    vf = self.get_file('fulltext')
    vf.add_lines(b'noeol', [], [b'line'])
    vf.add_lines(b'noeol2', [b'noeol'], [b'newline\n', b'line'], left_matching_blocks=[(0, 1, 1)])
    self.assertEqualDiff(b'newline\nline', vf.get_text(b'noeol2'))
    vf = self.get_file('delta')
    vf.add_lines(b'base', [], [b'line'])
    vf.add_lines(b'noeol', [b'base'], [b'prelude\n', b'line'])
    vf.add_lines(b'noeol2', [b'noeol'], [b'newline\n', b'line'], left_matching_blocks=[(1, 1, 1)])
    self.assertEqualDiff(b'newline\nline', vf.get_text(b'noeol2'))