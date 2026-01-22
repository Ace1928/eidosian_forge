from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def upload_file_robustly(self, relpath, mode=None):
    """Upload a file, clearing the way on the remote side.

        When doing a full upload, it may happen that a directory exists where
        we want to put our file.
        """
    self._force_clear(relpath)
    self.upload_file(relpath, relpath, mode)