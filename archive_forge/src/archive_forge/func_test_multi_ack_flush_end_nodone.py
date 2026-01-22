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
def test_multi_ack_flush_end_nodone(self):
    self._walker.lines[-1] = (None, None)
    self._walker.done_required = False
    self.assertNextEquals(TWO)
    self.assertNoAck()
    self.assertNextEquals(ONE)
    self._impl.ack(ONE)
    self.assertAck(ONE, b'common')
    self.assertNextEquals(THREE)
    self._impl.ack(THREE)
    self.assertAck(THREE, b'common')
    self._walker.wants_satisified = True
    self.assertNextEmpty()
    self.assertAcks([(THREE, b'ready'), (None, b'nak'), (THREE, b'')])
    self.assertTrue(self._walker.pack_sent)