from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def calculate_zoom_level_stats(super_tiles, load_data_func, rasterize_func, color_ranging_strategy='fullscan'):
    if color_ranging_strategy == 'fullscan':
        stats = []
        is_bool = False
        for super_tile in super_tiles:
            agg = _get_super_tile_min_max(super_tile, load_data_func, rasterize_func)
            super_tile['agg'] = agg
            if agg.dtype.kind == 'b':
                is_bool = True
            else:
                stats.append(np.nanmin(agg.data))
                stats.append(np.nanmax(agg.data))
        if is_bool:
            span = (0, 1)
        else:
            b = db.from_sequence(stats)
            span = dask.compute(b.min(), b.max())
        return (super_tiles, span)
    else:
        raise ValueError('Invalid color_ranging_strategy option')