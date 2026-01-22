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
@property
def firstbytes(self):
    """The first 256 bytes of the file. These can be used to
        parse the header to determine the file-format.
        """
    if self._firstbytes is None:
        self._read_first_bytes()
    return self._firstbytes