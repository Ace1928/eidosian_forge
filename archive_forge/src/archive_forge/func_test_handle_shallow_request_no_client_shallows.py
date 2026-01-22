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
def test_handle_shallow_request_no_client_shallows(self):
    self._handle_shallow_request([b'deepen 2\n'], [FOUR, FIVE])
    self.assertEqual({TWO, THREE}, self._walker.shallow)
    self.assertReceived([b'shallow ' + TWO, b'shallow ' + THREE])