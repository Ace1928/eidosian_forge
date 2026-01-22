import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_pop_last(self):
    compressor = self.compressor()
    text = b'some text\nfor the first entry\n'
    _, _, _, _ = compressor.compress(('key1',), [text], len(text), None)
    expected_lines = list(compressor.chunks)
    text = b'some text\nfor the second entry\n'
    _, _, _, _ = compressor.compress(('key2',), [text], len(text), None)
    compressor.pop_last()
    self.assertEqual(expected_lines, compressor.chunks)