import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def do_inconsistent_inserts(self, inconsistency_fatal):
    target = self.make_test_vf(True, dir='target', inconsistency_fatal=inconsistency_fatal)
    for x in range(2):
        source = self.make_source_with_b(x == 1, 'source%s' % x)
        target.insert_record_stream(source.get_record_stream([(b'b',)], 'unordered', False))