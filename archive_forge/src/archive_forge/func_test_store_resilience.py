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
def test_store_resilience(self):
    """Test if updating an existing stored object doesn't erase the
        object from the store.
        """
    test_object = make_object(Blob, data=b'data')
    self.store.add_object(test_object)
    test_object_id = test_object.id
    test_object.data = test_object.data + b'update'
    stored_test_object = self.store[test_object_id]
    self.assertNotEqual(test_object.id, stored_test_object.id)
    self.assertEqual(stored_test_object.id, test_object_id)