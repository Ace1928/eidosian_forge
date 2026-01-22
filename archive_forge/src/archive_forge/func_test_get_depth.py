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
def test_get_depth(self):
    self.assertEqual(0, self.store._get_depth(testobject.id))
    self.store.add_object(testobject)
    self.assertEqual(1, self.store._get_depth(testobject.id, get_parents=lambda x: []))
    parent = make_object(Blob, data=b'parent data')
    self.store.add_object(parent)
    self.assertEqual(2, self.store._get_depth(testobject.id, get_parents=lambda x: [parent.id] if x == testobject else []))