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
def get_ordering_vf(self, key_priority):
    builder = self.make_branch_builder('test')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', None))], revision_id=b'A')
    builder.build_snapshot([b'A'], [], revision_id=b'B')
    builder.build_snapshot([b'B'], [], revision_id=b'C')
    builder.build_snapshot([b'C'], [], revision_id=b'D')
    builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    vf = b.repository.inventories
    return versionedfile.OrderingVersionedFilesDecorator(vf, key_priority)