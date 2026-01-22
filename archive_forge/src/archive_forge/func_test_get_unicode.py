import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_get_unicode(self):
    cache = self.make_file_cache()
    self.assertIs(None, cache.get(self.unicode_text))