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
def test_get_loose_object(self):
    blob = make_object(Blob, data=b'foo')
    backend = _test_backend([blob])
    mat = re.search('^(..)(.{38})$', blob.id.decode('ascii'))
    output = b''.join(get_loose_object(self._req, backend, mat))
    self.assertEqual(blob.as_legacy_object(), output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('application/x-git-loose-object')
    self.assertTrue(self._req.cached)