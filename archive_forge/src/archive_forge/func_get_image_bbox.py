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
def get_image_bbox(self, name, transform=0):
    """Get rectangle occupied by image 'name'.

        'name' is either an item of the image list, or the referencing
        name string - elem[7] of the resp. item.
        Option 'transform' also returns the image transformation matrix.
        """
    CheckParent(self)
    doc = self.parent
    if doc.is_closed or doc.is_encrypted:
        raise ValueError('document closed or encrypted')
    inf_rect = Rect(1, 1, -1, -1)
    null_mat = Matrix()
    if transform:
        rc = (inf_rect, null_mat)
    else:
        rc = inf_rect
    if type(name) in (list, tuple):
        if not type(name[-1]) is int:
            raise ValueError('need item of full page image list')
        item = name
    else:
        imglist = [i for i in doc.get_page_images(self.number, True) if name == i[7]]
        if len(imglist) == 1:
            item = imglist[0]
        elif imglist == []:
            raise ValueError('bad image name')
        else:
            raise ValueError("found multiple images named '%s'." % name)
    xref = item[-1]
    if xref != 0 or transform is True:
        try:
            return self.get_image_rects(item, transform=transform)[0]
        except Exception:
            exception_info()
            return inf_rect
    pdf_page = self._pdf_page()
    val = JM_image_reporter(pdf_page)
    if not bool(val):
        return rc
    for v in val:
        if v[0] != item[-3]:
            continue
        q = Quad(v[1])
        bbox = q.rect
        if transform == 0:
            rc = bbox
            break
        hm = Matrix(util_hor_matrix(q.ll, q.lr))
        h = abs(q.ll - q.ul)
        w = abs(q.ur - q.ul)
        m0 = Matrix(1 / w, 0, 0, 1 / h, 0, 0)
        m = ~(hm * m0)
        rc = (bbox, m)
        break
    val = rc
    return val