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
def test_tempfile_in_loose_store(self):
    self.store.add_object(testobject)
    self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))
    for i in range(256):
        dirname = os.path.join(self.store_dir, '%02x' % i)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        fd, n = tempfile.mkstemp(prefix='tmp_obj_', dir=dirname)
        os.close(fd)
    self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))