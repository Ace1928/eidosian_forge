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
def test_get_texts_eol_variation(self):
    vf = self.get_file()
    sample_text_nl = [b'line\n']
    sample_text_no_nl = [b'line']
    versions = []
    version_lines = {}
    parents = []
    for i in range(4):
        version = b'v%d' % i
        if i % 2:
            lines = sample_text_nl
        else:
            lines = sample_text_no_nl
        vf.add_lines(version, parents, lines, left_matching_blocks=[(0, 0, 1)])
        parents = [version]
        versions.append(version)
        version_lines[version] = lines
    vf.check()
    vf.get_texts(versions)
    vf.get_texts(reversed(versions))