from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def create_sub_tiles(data_array, level, tile_info, output_path, post_render_func=None):
    _create_dir(output_path)
    tile_def = MercatorTileDefinition(x_range=tile_info['x_range'], y_range=tile_info['y_range'], tile_size=256)
    if output_path.startswith('s3:'):
        renderer = S3TileRenderer(tile_def, output_location=output_path, post_render_func=post_render_func)
    else:
        renderer = FileSystemTileRenderer(tile_def, output_location=output_path, post_render_func=post_render_func)
    return renderer.render(data_array, level=level)