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
def test_known_graph_with_fallbacks(self):
    f = self.get_versionedfiles('files')
    if not self.graph:
        raise TestNotApplicable('ancestry info only relevant with graph.')
    if getattr(f, 'add_fallback_versioned_files', None) is None:
        raise TestNotApplicable("%s doesn't support fallbacks" % (f.__class__.__name__,))
    key_a = self.get_simple_key(b'a')
    key_b = self.get_simple_key(b'b')
    key_c = self.get_simple_key(b'c')
    g = self.get_versionedfiles('fallback')
    g.add_lines(key_a, [], [b'\n'])
    f.add_fallback_versioned_files(g)
    f.add_lines(key_b, [key_a], [b'\n'])
    f.add_lines(key_c, [key_a, key_b], [b'\n'])
    kg = f.get_known_graph_ancestry([key_c])
    self.assertEqual([key_a, key_b, key_c], list(kg.topo_sort()))