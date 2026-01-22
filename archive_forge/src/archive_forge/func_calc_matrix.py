import io
import math
import os
import typing
import weakref
def calc_matrix(sr, tr, keep=True, rotate=0):
    """Calculate transformation matrix from source to target rect.

        Notes:
            The product of four matrices in this sequence: (1) translate correct
            source corner to origin, (2) rotate, (3) scale, (4) translate to
            target's top-left corner.
        Args:
            sr: source rect in PDF (!) coordinate system
            tr: target rect in PDF coordinate system
            keep: whether to keep source ratio of width to height
            rotate: rotation angle in degrees
        Returns:
            Transformation matrix.
        """
    smp = (sr.tl + sr.br) / 2.0
    tmp = (tr.tl + tr.br) / 2.0
    m = fitz.Matrix(1, 0, 0, 1, -smp.x, -smp.y) * fitz.Matrix(rotate)
    sr1 = sr * m
    fw = tr.width / sr1.width
    fh = tr.height / sr1.height
    if keep:
        fw = fh = min(fw, fh)
    m *= fitz.Matrix(fw, fh)
    m *= fitz.Matrix(1, 0, 0, 1, tmp.x, tmp.y)
    return fitz.JM_TUPLE(m)