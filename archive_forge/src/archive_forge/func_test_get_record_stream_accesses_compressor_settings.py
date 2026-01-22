import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_get_record_stream_accesses_compressor_settings(self):
    vf = self.make_test_vf(True, dir='source')
    vf.add_lines((b'a',), (), [b'lines\n'])
    vf.writer.end()
    vf._max_bytes_to_index = 1234
    record = next(vf.get_record_stream([(b'a',)], 'unordered', True))
    self.assertEqual(dict(max_bytes_to_index=1234), record._manager._get_compressor_settings())