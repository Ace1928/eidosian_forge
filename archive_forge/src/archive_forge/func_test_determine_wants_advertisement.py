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
def test_determine_wants_advertisement(self):
    self._walker.proto.set_output([None])
    heads = {b'refs/heads/ref4': FOUR, b'refs/heads/ref5': FIVE, b'refs/heads/tag6': SIX}
    self._repo.refs._update(heads)
    self._repo.refs._update_peeled(heads)
    self._repo.refs._update_peeled({b'refs/heads/tag6': FIVE})
    self._walker.determine_wants(heads)
    lines = []
    while True:
        line = self._walker.proto.get_received_line()
        if line is None:
            break
        if b'\x00' in line:
            line = line[:line.index(b'\x00')]
        lines.append(line.rstrip())
    self.assertEqual([FOUR + b' refs/heads/ref4', FIVE + b' refs/heads/ref5', FIVE + b' refs/heads/tag6^{}', SIX + b' refs/heads/tag6'], sorted(lines))
    for i, line in enumerate(lines):
        if line.endswith(b' refs/heads/tag6'):
            self.assertEqual(FIVE + b' refs/heads/tag6^{}', lines[i + 1])