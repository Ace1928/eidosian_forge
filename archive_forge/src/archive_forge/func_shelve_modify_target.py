import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_modify_target(self, file_id):
    """Shelve a change of symlink target.

        :param file_id: The file id of the symlink which changed target.
        :param new_target: The target that the symlink should have due
            to shelving.
        """
    new_path = self.target_tree.id2path(file_id)
    new_target = self.target_tree.get_symlink_target(new_path)
    w_trans_id = self.work_transform.trans_id_file_id(file_id)
    self.work_transform.delete_contents(w_trans_id)
    self.work_transform.create_symlink(new_target, w_trans_id)
    old_path = self.work_tree.id2path(file_id)
    old_target = self.work_tree.get_symlink_target(old_path)
    s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
    self.shelf_transform.delete_contents(s_trans_id)
    self.shelf_transform.create_symlink(old_target, s_trans_id)