import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_bad_safename_set(self):
    safename = self.prefix_safename
    cache = AtomicFileCache(self.cache_dir, safename)
    self.assertRaises(ValueError, cache.set, 'key', b'value')