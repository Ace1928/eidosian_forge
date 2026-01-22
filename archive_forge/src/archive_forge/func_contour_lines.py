import numpy as np
import pandas as pd
from ..doctools import document
from .density import get_var_type, kde
from .stat import stat
def contour_lines(X, Y, Z, levels):
    """
    Calculate contour lines
    """
    from contourpy import contour_generator
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    zmin, zmax = (Z.min(), Z.max())
    cgen = contour_generator(X, Y, Z, name='mpl2014', corner_mask=False, chunk_size=0)
    if isinstance(levels, int):
        from mizani.breaks import breaks_extended
        levels = breaks_extended(n=levels)((zmin, zmax))
    segments = []
    piece_ids = []
    level_values = []
    start_pid = 1
    for level in levels:
        vertices, *_ = cgen.create_contour(level)
        for pid, piece in enumerate(vertices, start=start_pid):
            n = len(piece)
            segments.append(piece)
            piece_ids.append(np.repeat(pid, n))
            level_values.append(np.repeat(level, n))
            start_pid = pid + 1
    if segments:
        x, y = np.vstack(segments).T
        piece = np.hstack(piece_ids)
        level = np.hstack(level_values)
    else:
        x, y = ([], [])
        piece = []
        level = []
    data = pd.DataFrame({'x': x, 'y': y, 'level': level, 'piece': piece})
    return data