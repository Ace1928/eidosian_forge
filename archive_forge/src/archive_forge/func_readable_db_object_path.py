from gitdb.db.base import (
from gitdb.exc import (
from gitdb.stream import (
from gitdb.base import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.utils.encoding import force_bytes
import tempfile
import os
import sys
def readable_db_object_path(self, hexsha):
    """
        :return: readable object path to the object identified by hexsha
        :raise BadObject: If the object file does not exist"""
    try:
        return self._hexsha_to_file[hexsha]
    except KeyError:
        pass
    path = self.db_path(self.object_path(hexsha))
    if exists(path):
        self._hexsha_to_file[hexsha] = path
        return path
    raise BadObject(hexsha)