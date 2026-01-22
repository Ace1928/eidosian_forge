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
def test_add_follows_left_matching_blocks(self):
    """If we change left_matching_blocks, delta changes

        Note: There are multiple correct deltas in this case, because
        we start with 1 "a" and we get 3.
        """
    vf = self.get_file()
    if isinstance(vf, WeaveFile):
        raise TestSkipped('WeaveFile ignores left_matching_blocks')
    vf.add_lines(b'1', [], [b'a\n'])
    vf.add_lines(b'2', [b'1'], [b'a\n', b'a\n', b'a\n'], left_matching_blocks=[(0, 0, 1), (1, 3, 0)])
    self.assertEqual([b'a\n', b'a\n', b'a\n'], vf.get_lines(b'2'))
    vf.add_lines(b'3', [b'1'], [b'a\n', b'a\n', b'a\n'], left_matching_blocks=[(0, 2, 1), (1, 3, 0)])
    self.assertEqual([b'a\n', b'a\n', b'a\n'], vf.get_lines(b'3'))