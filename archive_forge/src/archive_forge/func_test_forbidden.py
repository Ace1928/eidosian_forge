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
def test_forbidden(self):
    self._req.cache_forever()
    message = 'Something not found'
    self.assertEqual(message.encode('ascii'), self._req.forbidden(message))
    self.assertEqual(HTTP_FORBIDDEN, self._status)
    self.assertEqual({('Content-Type', 'text/plain')}, set(self._headers))