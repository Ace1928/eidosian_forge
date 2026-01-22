import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_get_non_string_key(self):
    cache = self.make_file_cache()
    self.assertRaises(TypeError, cache.get, 42)