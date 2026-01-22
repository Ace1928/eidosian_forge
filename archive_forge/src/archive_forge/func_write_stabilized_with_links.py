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
@staticmethod
def write_stabilized_with_links(contentfn, rectfn, user_css=None, em=12, positionfn=None, pagefn=None, archive=None, add_header_ids=True):
    stream = io.BytesIO()
    writer = DocumentWriter(stream)
    positions = []

    def positionfn2(position):
        positions.append(position)
        if positionfn:
            positionfn(position)
    Story.write_stabilized(writer, contentfn, rectfn, user_css, em, positionfn2, pagefn, archive, add_header_ids)
    writer.close()
    stream.seek(0)
    return Story.add_pdf_links(stream, positions)