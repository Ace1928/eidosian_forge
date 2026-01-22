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
def switch_layer(self, config, as_default=0):
    """Activate an OC layer."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    cfgs = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('OCProperties'), PDF_NAME('Configs'))
    if not mupdf.pdf_is_array(cfgs) or not mupdf.pdf_array_len(cfgs):
        if config < 1:
            return
        raise ValueError(MSG_BAD_OC_LAYER)
    if config < 0:
        return
    mupdf.pdf_select_layer_config(pdf, config)
    if as_default:
        mupdf.pdf_set_layer_config_as_default(pdf)
        mupdf.ll_pdf_read_ocg(pdf.m_internal)