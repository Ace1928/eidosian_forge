import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def delete_shelf(self, shelf_id):
    """Delete the shelved changes for a given id.

        :param shelf_id: id of the shelved changes to delete.
        """
    filename = self.get_shelf_filename(shelf_id)
    self.transport.delete(filename)