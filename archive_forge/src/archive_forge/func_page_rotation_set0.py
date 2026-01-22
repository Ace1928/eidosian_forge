import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def page_rotation_set0(page):
    """Nullify page rotation.

    To correctly detect tables, page rotation must be zero.
    This function performs the necessary adjustments and returns information
    for reverting this changes.
    """
    mediabox = page.mediabox
    rot = page.rotation
    mb = page.mediabox
    if rot == 90:
        mat0 = Matrix(1, 0, 0, 1, mb.y1 - mb.x1 - mb.x0 - mb.y0, 0)
    elif rot == 270:
        mat0 = Matrix(1, 0, 0, 1, 0, mb.x1 - mb.y1 - mb.y0 - mb.x0)
    else:
        mat0 = Matrix(1, 0, 0, 1, -2 * mb.x0, -2 * mb.y0)
    mat = mat0 * page.derotation_matrix
    cmd = b'%g %g %g %g %g %g cm ' % tuple(mat)
    xref = TOOLS._insert_contents(page, cmd, 0)
    if rot in (90, 270):
        x0, y0, x1, y1 = mb
        mb.x0 = y0
        mb.y0 = x0
        mb.x1 = y1
        mb.y1 = x1
        page.set_mediabox(mb)
    page.set_rotation(0)
    doc = page.parent
    pno = page.number
    page = doc[pno]
    return (page, xref, rot, mediabox)