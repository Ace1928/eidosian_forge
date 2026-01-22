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
def test_multi_ack_flush(self):
    self._walker.lines = [(b'have', TWO), (None, None), (b'have', ONE), (b'have', THREE), (b'done', None), (None, None)]
    self.assertNextEquals(TWO)
    self.assertNoAck()
    self.assertNextEquals(ONE)
    self.assertNak()
    self._impl.ack(ONE)
    self.assertAck(ONE, b'common')
    self.assertNextEquals(THREE)
    self._impl.ack(THREE)
    self.assertAck(THREE, b'common')
    self._walker.wants_satisified = True
    self.assertNextEquals(None)
    self.assertNextEmpty()
    self.assertAcks([(THREE, b'ready'), (None, b'nak'), (THREE, b'')])