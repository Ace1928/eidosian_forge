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
def test_prefix_mapper(self):
    mapper = versionedfile.PrefixMapper()
    self.assertEqual('file-id', mapper.map((b'file-id', b'revision-id')))
    self.assertEqual('new-id', mapper.map((b'new-id', b'revision-id')))
    self.assertEqual((b'file-id',), mapper.unmap('file-id'))
    self.assertEqual((b'new-id',), mapper.unmap('new-id'))