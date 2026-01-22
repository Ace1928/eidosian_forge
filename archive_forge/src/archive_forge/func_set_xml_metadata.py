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
def set_xml_metadata(self, metadata):
    """Store XML document level metadata."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    if not root.m_internal:
        RAISEPY(MSG_BAD_PDFROOT, JM_Exc_FileDataError)
    res = mupdf.fz_new_buffer_from_copied_data(metadata.encode('utf-8'))
    xml = mupdf.pdf_dict_get(root, PDF_NAME('Metadata'))
    if xml.m_internal:
        JM_update_stream(pdf, xml, res, 0)
    else:
        xml = mupdf.pdf_add_stream(pdf, res, mupdf.PdfObj(), 0)
        mupdf.pdf_dict_put(xml, PDF_NAME('Type'), PDF_NAME('Metadata'))
        mupdf.pdf_dict_put(xml, PDF_NAME('Subtype'), PDF_NAME('XML'))
        mupdf.pdf_dict_put(root, PDF_NAME('Metadata'), xml)