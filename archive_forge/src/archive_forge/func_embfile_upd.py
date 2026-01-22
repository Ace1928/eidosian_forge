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
def embfile_upd(self, item: typing.Union[int, str], buffer_: OptBytes=None, filename: OptStr=None, ufilename: OptStr=None, desc: OptStr=None) -> None:
    """Change an item of the EmbeddedFiles array.

        Notes:
            Only provided parameters are changed. If all are omitted,
            the method is a no-op.
        Args:
            item: number or name of item.
            buffer_: (binary data) the new file content.
            filename: (str) the new file name.
            ufilename: (unicode) the new filen ame.
            desc: (str) the new description.
        """
    idx = self._embeddedFileIndex(item)
    xref = self._embfile_upd(idx, buffer_=buffer_, filename=filename, ufilename=ufilename, desc=desc)
    date = get_pdf_now()
    self.xref_set_key(xref, 'Params/ModDate', get_pdf_str(date))
    return xref