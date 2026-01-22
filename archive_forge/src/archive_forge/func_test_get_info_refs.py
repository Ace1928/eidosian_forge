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
def test_get_info_refs(self):
    self._environ['wsgi.input'] = BytesIO(b'foo')
    self._environ['QUERY_STRING'] = 'service=git-upload-pack'

    class Backend:

        def open_repository(self, url):
            return None
    mat = re.search('.*', '/git-upload-pack')
    handler_output = b''.join(get_info_refs(self._req, Backend(), mat))
    write_output = self._output.getvalue()
    self.assertEqual(b'001e# service=git-upload-pack\n0000handled input: ', write_output)
    self.assertEqual(b'', handler_output)
    self.assertTrue(self._handler.advertise_refs)
    self.assertTrue(self._handler.stateless_rpc)
    self.assertFalse(self._req.cached)