import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
def get_local_filename(self):
    """get_local_filename()
        If the filename is an existing file on this filesystem, return
        that. Otherwise a temporary file is created on the local file
        system which can be used by the format to read from or write to.
        """
    if self._uri_type == URI_FILENAME:
        return self._filename
    else:
        if self.extension is not None:
            ext = self.extension
        else:
            ext = os.path.splitext(self._filename)[1]
        fd, self._filename_local = tempfile.mkstemp(ext, 'imageio_')
        os.close(fd)
        if self.mode.io_mode == IOMode.read:
            with open(self._filename_local, 'wb') as file:
                shutil.copyfileobj(self.get_file(), file)
        return self._filename_local