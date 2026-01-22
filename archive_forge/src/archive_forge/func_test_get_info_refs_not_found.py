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
def test_get_info_refs_not_found(self):
    self._environ['QUERY_STRING'] = ''
    objects = []
    refs = {}
    backend = _test_backend(objects, refs=refs)
    mat = re.search('info/refs', '/foo/info/refs')
    self.assertEqual([b'No git repository was found at /foo'], list(get_info_refs(self._req, backend, mat)))
    self.assertEqual(HTTP_NOT_FOUND, self._status)
    self.assertContentTypeEquals('text/plain')