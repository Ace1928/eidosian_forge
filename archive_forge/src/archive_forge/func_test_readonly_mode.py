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
def test_readonly_mode(self):
    t = self.get_transport()
    factory = self.get_factory()
    vf = factory('id', t, 511, create=True, access_mode='w')
    vf = factory('id', t, access_mode='r')
    self.assertRaises(errors.ReadOnlyError, vf.add_lines, b'base', [], [])
    self.assertRaises(errors.ReadOnlyError, vf.add_lines_with_ghosts, b'base', [], [])