import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def check_shelve_change_kind(self, creator):
    self.assertFileEqual(b'bar', 'tree/foo')
    s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    self.assertEqual('directory', creator.shelf_transform._new_contents[s_trans_id])