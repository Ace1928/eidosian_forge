import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_lookup_tree(self):
    o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad')[1]
    self.assertIsInstance(self.store[o_id], Tree)
    o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad/bd')[1]
    self.assertIsInstance(self.store[o_id], Tree)
    o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad/bd/')[1]
    self.assertIsInstance(self.store[o_id], Tree)