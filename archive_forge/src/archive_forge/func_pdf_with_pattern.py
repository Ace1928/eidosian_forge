import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def pdf_with_pattern(pattern=None):
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 100, 100)
    context = Context(surface)
    if pattern is not None:
        context.set_source(pattern)
    context.paint()
    surface.finish()
    return file_obj.getvalue()