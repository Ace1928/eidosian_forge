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
def test_add_lines_no_key_generates_chk_key(self):
    files = self.get_versionedfiles()
    adds = self.get_diamond_files(files, nokeys=True)
    results = []
    for add in adds:
        self.assertEqual(3, len(add))
        results.append(add[:2])
    if self.key_length == 1:
        self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)
        self.assertEqual({(b'sha1:00e364d235126be43292ab09cb4686cf703ddc17',), (b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44',), (b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1',), (b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f',), (b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d',)}, files.keys())
    elif self.key_length == 2:
        self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)
        self.assertEqual({(b'FileA', b'sha1:00e364d235126be43292ab09cb4686cf703ddc17'), (b'FileA', b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44'), (b'FileA', b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1'), (b'FileA', b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f'), (b'FileA', b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d'), (b'FileB', b'sha1:00e364d235126be43292ab09cb4686cf703ddc17'), (b'FileB', b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44'), (b'FileB', b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1'), (b'FileB', b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f'), (b'FileB', b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d')}, files.keys())