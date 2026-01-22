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
def test_tree_changes(self):
    blob_a1 = make_object(Blob, data=b'a1')
    blob_a2 = make_object(Blob, data=b'a2')
    blob_b = make_object(Blob, data=b'b')
    for blob in [blob_a1, blob_a2, blob_b]:
        self.store.add_object(blob)
    blobs_1 = [(b'a', blob_a1.id, 33188), (b'b', blob_b.id, 33188)]
    tree1_id = commit_tree(self.store, blobs_1)
    blobs_2 = [(b'a', blob_a2.id, 33188), (b'b', blob_b.id, 33188)]
    tree2_id = commit_tree(self.store, blobs_2)
    change_a = ((b'a', b'a'), (33188, 33188), (blob_a1.id, blob_a2.id))
    self.assertEqual([change_a], list(self.store.tree_changes(tree1_id, tree2_id)))
    self.assertEqual([change_a, ((b'b', b'b'), (33188, 33188), (blob_b.id, blob_b.id))], list(self.store.tree_changes(tree1_id, tree2_id, want_unchanged=True)))