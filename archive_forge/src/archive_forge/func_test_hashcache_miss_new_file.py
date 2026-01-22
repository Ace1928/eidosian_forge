import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def test_hashcache_miss_new_file(self):
    """A new file gives the right sha1 but misses"""
    hc = self.make_hashcache()
    hc.put_file('foo', b'hello')
    self.assertEqual(hc.get_sha1('foo'), sha1(b'hello'))
    self.assertEqual(hc.miss_count, 1)
    self.assertEqual(hc.hit_count, 0)
    self.assertEqual(hc.get_sha1('foo'), sha1(b'hello'))
    self.assertEqual(hc.miss_count, 2)
    self.assertEqual(hc.hit_count, 0)