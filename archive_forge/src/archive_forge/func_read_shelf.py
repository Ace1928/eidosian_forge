import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def read_shelf(self, shelf_id):
    """Return the file associated with a shelf_id for reading.

        :param shelf_id: The id of the shelf to retrive the file for.
        """
    filename = self.get_shelf_filename(shelf_id)
    try:
        return open(self.transport.local_abspath(filename), 'rb')
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        raise NoSuchShelfId(shelf_id)