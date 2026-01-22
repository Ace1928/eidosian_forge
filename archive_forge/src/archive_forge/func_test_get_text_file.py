import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
def test_get_text_file(self):
    backend = _test_backend([], named_files={'description': b'foo'})
    mat = re.search('.*', 'description')
    output = b''.join(get_text_file(self._req, backend, mat))
    self.assertEqual(b'foo', output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('text/plain')
    self.assertFalse(self._req.cached)