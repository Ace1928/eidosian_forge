from __future__ import annotations
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis
from datashader.tiles import render_tiles
from datashader.tiles import gen_super_tiles
from datashader.tiles import _get_super_tile_min_max
from datashader.tiles import calculate_zoom_level_stats
from datashader.tiles import MercatorTileDefinition
import numpy as np
import pandas as pd
def mock_rasterize_func(df, x_range, y_range, height, width):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=height, plot_width=width)
    agg = cvs.points(df, 'x', 'y')
    return agg