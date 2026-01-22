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
def test_not_found(self):
    self._req.cache_forever()
    message = 'Something not found'
    self.assertEqual(message.encode('ascii'), self._req.not_found(message))
    self.assertEqual(HTTP_NOT_FOUND, self._status)
    self.assertEqual({('Content-Type', 'text/plain')}, set(self._headers))