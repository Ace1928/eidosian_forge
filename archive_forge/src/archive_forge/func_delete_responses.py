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
def delete_responses(self):
    """Delete 'Popup' and responding annotations."""
    CheckParent(self)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    page = mupdf.pdf_annot_page(annot)
    while 1:
        irt_annot = JM_find_annot_irt(annot)
        if not irt_annot.m_internal:
            break
        mupdf.pdf_delete_annot(page, irt_annot)
    mupdf.pdf_dict_del(annot_obj, PDF_NAME('Popup'))
    annots = mupdf.pdf_dict_get(page.obj(), PDF_NAME('Annots'))
    n = mupdf.pdf_array_len(annots)
    found = 0
    for i in range(n - 1, -1, -1):
        o = mupdf.pdf_array_get(annots, i)
        p = mupdf.pdf_dict_get(o, PDF_NAME('Parent'))
        if not o.m_internal:
            continue
        if not mupdf.pdf_objcmp(p, annot_obj):
            mupdf.pdf_array_delete(annots, i)
            found = 1
    if found:
        mupdf.pdf_dict_put(page.obj(), PDF_NAME('Annots'), annots)