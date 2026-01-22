from toolz import memoize
import numpy as np
from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeometryLike
from datashader.utils import ngjit
@ngjit
@expand_aggs_and_cols
def extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets0, offsets1, offsets2, eligible_inds, *aggs_and_cols):
    max_edges = 0
    if len(offsets0) > 1:
        for i in eligible_inds:
            if missing[i]:
                continue
            polygon_inds = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            for j in range(len(polygon_inds) - 1):
                start = offsets2[polygon_inds[j]]
                stop = offsets2[polygon_inds[j + 1]]
                max_edges = max(max_edges, (stop - start - 2) // 2)
    xs = np.full((max_edges, 2), np.nan, dtype=np.float32)
    ys = np.full((max_edges, 2), np.nan, dtype=np.float32)
    yincreasing = np.zeros(max_edges, dtype=np.int8)
    eligible = np.ones(max_edges, dtype=np.int8)
    for i in eligible_inds:
        if missing[i]:
            continue
        polygon_inds = offsets1[offsets0[i]:offsets0[i + 1] + 1]
        for j in range(len(polygon_inds) - 1):
            start = polygon_inds[j]
            stop = polygon_inds[j + 1]
            draw_polygon(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, offsets2[start:stop + 1], 1, values, xs, ys, yincreasing, eligible, *aggs_and_cols)