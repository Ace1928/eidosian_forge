import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_three_nosha_delta(self):
    compressor = self.compressor()
    text = b'strange\ncommon very very long line\nwith some extra text\n'
    sha1_1, _, _, _ = compressor.compress(('label',), [text], len(text), None)
    text = b'different\nmoredifferent\nand then some more\n'
    sha1_2, _, _, _ = compressor.compress(('newlabel',), [text], len(text), None)
    expected_lines = list(compressor.chunks)
    text = b'new\ncommon very very long line\nwith some extra text\ndifferent\nmoredifferent\nand then some more\n'
    sha1_3, start_point, end_point, _ = compressor.compress(('label3',), [text], len(text), None)
    self.assertEqual(sha_string(text), sha1_3)
    expected_lines.extend([b'd\x0c', b'_\x04new\n', b'\x91\n0\x91<+'])
    self.assertEqualDiffEncoded(expected_lines, compressor.chunks)
    self.assertEqual(sum(map(len, expected_lines)), end_point)