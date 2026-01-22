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
def write_with_links(self, rectfn, positionfn=None, pagefn=None):
    stream = io.BytesIO()
    writer = DocumentWriter(stream)
    positions = []

    def positionfn2(position):
        positions.append(position)
        if positionfn:
            positionfn(position)
    self.write(writer, rectfn, positionfn=positionfn2, pagefn=pagefn)
    writer.close()
    stream.seek(0)
    return Story.add_pdf_links(stream, positions)