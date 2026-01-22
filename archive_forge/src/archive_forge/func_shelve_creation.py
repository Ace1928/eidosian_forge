import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_creation(self, file_id):
    """Shelve creation of a file.

        This handles content and inventory id.
        :param file_id: The file_id of the file to shelve creation of.
        """
    kind, name, parent, versioned = self.creation[file_id]
    version = not versioned[0]
    self._shelve_creation(self.work_tree, file_id, self.work_transform, self.shelf_transform, kind, name, parent, version)