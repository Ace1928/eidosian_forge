import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
def test_has_capability(self):
    self.assertRaises(GitProtocolError, self._handler.has_capability, b'cap')
    caps = self._handler.capabilities()
    self._handler.set_client_capabilities(caps)
    for cap in caps:
        self.assertTrue(self._handler.has_capability(cap))
    self.assertFalse(self._handler.has_capability(b'capxxx'))