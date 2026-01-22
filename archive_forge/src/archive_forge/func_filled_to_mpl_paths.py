from __future__ import annotations
from typing import TYPE_CHECKING, cast
import matplotlib.path as mpath
import numpy as np
from contourpy import FillType, LineType
from contourpy.array import codes_from_offsets
def filled_to_mpl_paths(filled: FillReturn, fill_type: FillType) -> list[mpath.Path]:
    if fill_type in (FillType.OuterCode, FillType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*filled) if points is not None]
    elif fill_type in (FillType.OuterOffset, FillType.ChunkCombinedOffset):
        paths = [mpath.Path(points, codes_from_offsets(offsets)) for points, offsets in zip(*filled) if points is not None]
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        paths = []
        for points, codes, outer_offsets in zip(*filled):
            if points is None:
                continue
            points = np.split(points, outer_offsets[1:-1])
            codes = np.split(codes, outer_offsets[1:-1])
            paths += [mpath.Path(p, c) for p, c in zip(points, codes)]
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        paths = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is None:
                continue
            for i in range(len(outer_offsets) - 1):
                offs = offsets[outer_offsets[i]:outer_offsets[i + 1] + 1]
                pts = points[offs[0]:offs[-1]]
                paths += [mpath.Path(pts, codes_from_offsets(offs - offs[0]))]
    else:
        raise RuntimeError(f'Conversion of FillType {fill_type} to MPL Paths is not implemented')
    return paths