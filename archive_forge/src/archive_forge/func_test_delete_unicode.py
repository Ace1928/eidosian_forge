import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_delete_unicode(self):
    cache = self.make_file_cache()
    cache.set(self.unicode_text, b'value')
    cache.delete(self.unicode_text)
    self.assertIs(None, cache.get(self.unicode_text))