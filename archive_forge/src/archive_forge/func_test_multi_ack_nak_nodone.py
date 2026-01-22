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
def test_multi_ack_nak_nodone(self):
    self._walker.done_required = False
    self.assertNextEquals(TWO)
    self.assertNoAck()
    self.assertNextEquals(ONE)
    self.assertNoAck()
    self.assertNextEquals(THREE)
    self.assertNoAck()
    self.assertFalse(self._walker.pack_sent)
    self.assertNextEquals(None)
    self.assertNextEmpty()
    self.assertTrue(self._walker.pack_sent)
    self.assertNak()
    self.assertNextEmpty()