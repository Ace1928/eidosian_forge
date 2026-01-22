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
def read_pkt_line(self):
    if self._output:
        data = self._output.pop(0)
        if data is not None:
            return data.rstrip() + b'\n'
        else:
            return None
    else:
        raise HangupException