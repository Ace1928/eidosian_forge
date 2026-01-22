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
def test_loose_compression_level(self):
    alternate_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, alternate_dir)
    alternate_store = DiskObjectStore(alternate_dir, loose_compression_level=6)
    b2 = make_object(Blob, data=b'yummy data')
    alternate_store.add_object(b2)