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
def doMerge(self, base, a, b, mp):
    from textwrap import dedent

    def addcrlf(x):
        return x + b'\n'
    w = self.get_file()
    w.add_lines(b'text0', [], list(map(addcrlf, base)))
    w.add_lines(b'text1', [b'text0'], list(map(addcrlf, a)))
    w.add_lines(b'text2', [b'text0'], list(map(addcrlf, b)))
    self.log_contents(w)
    self.log('merge plan:')
    p = list(w.plan_merge(b'text1', b'text2'))
    for state, line in p:
        if line:
            self.log('%12s | %s' % (state, line[:-1]))
    self.log('merge:')
    mt = BytesIO()
    mt.writelines(w.weave_merge(p))
    mt.seek(0)
    self.log(mt.getvalue())
    mp = list(map(addcrlf, mp))
    self.assertEqual(mt.readlines(), mp)