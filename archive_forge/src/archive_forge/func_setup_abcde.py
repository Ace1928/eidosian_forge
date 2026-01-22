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
def setup_abcde(self):
    self.vf1.add_lines((b'root', b'A'), [], [b'a'])
    self.vf1.add_lines((b'root', b'B'), [(b'root', b'A')], [b'b'])
    self.vf2.add_lines((b'root', b'C'), [], [b'c'])
    self.vf2.add_lines((b'root', b'D'), [(b'root', b'C')], [b'd'])
    self.plan_merge_vf.add_lines((b'root', b'E:'), [(b'root', b'B'), (b'root', b'D')], [b'e'])