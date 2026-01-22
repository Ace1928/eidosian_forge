import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_rename(self, file_id):
    """Shelve a file rename.

        :param file_id: The file id of the file to shelve the renaming of.
        """
    names, parents = self.renames[file_id]
    w_trans_id = self.work_transform.trans_id_file_id(file_id)
    work_parent = self.work_transform.trans_id_file_id(parents[0])
    self.work_transform.adjust_path(names[0], work_parent, w_trans_id)
    s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
    shelf_parent = self.shelf_transform.trans_id_file_id(parents[1])
    self.shelf_transform.adjust_path(names[1], shelf_parent, s_trans_id)