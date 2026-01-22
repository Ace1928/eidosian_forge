import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def make_file_cache(self):
    """Make a FileCache-like object to be tested."""
    return self.file_cache_factory(self.cache_dir, safename)