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
def test_send_file(self):
    f = BytesIO(b'foobar')
    output = b''.join(send_file(self._req, f, 'some/thing'))
    self.assertEqual(b'foobar', output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('some/thing')
    self.assertTrue(f.closed)