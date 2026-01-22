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
def test_determine_wants(self):
    self._walker.proto.set_output([None])
    self.assertEqual([], self._walker.determine_wants({}))
    self.assertEqual(None, self._walker.proto.get_received_line())
    self._walker.proto.set_output([b'want ' + ONE + b' multi_ack', b'want ' + TWO, None])
    heads = {b'refs/heads/ref1': ONE, b'refs/heads/ref2': TWO, b'refs/heads/ref3': THREE}
    self._repo.refs._update(heads)
    self.assertEqual([ONE, TWO], self._walker.determine_wants(heads))
    self._walker.advertise_refs = True
    self.assertEqual([], self._walker.determine_wants(heads))
    self._walker.advertise_refs = False
    self._walker.proto.set_output([b'want ' + FOUR + b' multi_ack', None])
    self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)
    self._walker.proto.set_output([None])
    self.assertEqual([], self._walker.determine_wants(heads))
    self._walker.proto.set_output([b'want ' + ONE + b' multi_ack', b'foo', None])
    self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)
    self._walker.proto.set_output([b'want ' + FOUR + b' multi_ack', None])
    self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)