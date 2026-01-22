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
def test_add_lines_with_ghosts(self):
    vf = self.get_file()
    parent_id_unicode = 'b¿se'
    parent_id_utf8 = parent_id_unicode.encode('utf8')
    try:
        vf.add_lines_with_ghosts(b'notbxbfse', [parent_id_utf8], [])
    except NotImplementedError:
        self.assertRaises(NotImplementedError, vf.get_ancestry_with_ghosts, [b'foo'])
        self.assertRaises(NotImplementedError, vf.get_parents_with_ghosts, b'foo')
        return
    vf = self.reopen_file()
    self.assertEqual({b'notbxbfse'}, vf.get_ancestry(b'notbxbfse'))
    self.assertFalse(vf.has_version(parent_id_utf8))
    self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry_with_ghosts([b'notbxbfse']))
    self.assertEqual([parent_id_utf8], vf.get_parents_with_ghosts(b'notbxbfse'))
    vf.add_lines(parent_id_utf8, [], [])
    self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry([b'notbxbfse']))
    self.assertEqual({b'notbxbfse': (parent_id_utf8,)}, vf.get_parent_map([b'notbxbfse']))
    self.assertTrue(vf.has_version(parent_id_utf8))
    self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry_with_ghosts([b'notbxbfse']))
    self.assertEqual([parent_id_utf8], vf.get_parents_with_ghosts(b'notbxbfse'))