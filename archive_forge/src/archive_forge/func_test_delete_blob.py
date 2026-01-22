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
def test_delete_blob(self):
    new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'ad/bd/c', None, None)])
    self.assertEqual(set(new_tree), {b'a', b'ad', b'c'})
    ad_tree = self.store[new_tree[b'ad'][1]]
    self.assertEqual(set(ad_tree), {b'b', b'c'})