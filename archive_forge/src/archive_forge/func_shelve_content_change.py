import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_content_change(self, file_id):
    """Shelve a kind change or binary file content change.

        :param file_id: The file id of the file to shelve the content change
            of.
        """
    self._content_from_tree(self.work_transform, self.target_tree, file_id)
    self._content_from_tree(self.shelf_transform, self.work_tree, file_id)