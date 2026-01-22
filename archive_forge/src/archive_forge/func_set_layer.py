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
def set_layer(self, config, basestate=None, on=None, off=None, rbgroups=None, locked=None):
    """Set the PDF keys /ON, /OFF, /RBGroups of an OC layer."""
    if self.is_closed:
        raise ValueError('document closed')
    ocgs = set(self.get_ocgs().keys())
    if ocgs == set():
        raise ValueError('document has no optional content')
    if on:
        if type(on) not in (list, tuple):
            raise ValueError("bad type: 'on'")
        s = set(on).difference(ocgs)
        if s != set():
            raise ValueError("bad OCGs in 'on': %s" % s)
    if off:
        if type(off) not in (list, tuple):
            raise ValueError("bad type: 'off'")
        s = set(off).difference(ocgs)
        if s != set():
            raise ValueError("bad OCGs in 'off': %s" % s)
    if locked:
        if type(locked) not in (list, tuple):
            raise ValueError("bad type: 'locked'")
        s = set(locked).difference(ocgs)
        if s != set():
            raise ValueError("bad OCGs in 'locked': %s" % s)
    if rbgroups:
        if type(rbgroups) not in (list, tuple):
            raise ValueError("bad type: 'rbgroups'")
        for x in rbgroups:
            if not type(x) in (list, tuple):
                raise ValueError("bad RBGroup '%s'" % x)
            s = set(x).difference(ocgs)
            if s != set():
                raise ValueError('bad OCGs in RBGroup: %s' % s)
    if basestate:
        basestate = str(basestate).upper()
        if basestate == 'UNCHANGED':
            basestate = 'Unchanged'
        if basestate not in ('ON', 'OFF', 'Unchanged'):
            raise ValueError("bad 'basestate'")
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    ocp = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('OCProperties'))
    if not ocp.m_internal:
        return
    if config == -1:
        obj = mupdf.pdf_dict_get(ocp, PDF_NAME('D'))
    else:
        obj = mupdf.pdf_array_get(mupdf.pdf_dict_get(ocp, PDF_NAME('Configs')), config)
    if not obj.m_internal:
        raise ValueError(MSG_BAD_OC_CONFIG)
    JM_set_ocg_arrays(obj, basestate, on, off, rbgroups, locked)
    mupdf.ll_pdf_read_ocg(pdf.m_internal)