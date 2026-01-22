import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def pil_tobytes(self, *args, **kwargs):
    """Convert to binary image stream using pillow.

        Args are passed to Pillow's Image.save method, see their documentation.
        Use instead of 'tobytes' when other output formats are needed.
        """
    from io import BytesIO
    bytes_out = BytesIO()
    self.pil_save(bytes_out, *args, **kwargs)
    return bytes_out.getvalue()