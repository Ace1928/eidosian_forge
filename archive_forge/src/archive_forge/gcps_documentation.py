import json
import click
from cligj import (
import rasterio
import rasterio.crs
from rasterio.rio import options
from rasterio.warp import transform_geom
Print GeoJSON representations of a dataset's control points.

    Each ground control point is represented as a GeoJSON feature. The
    'properties' member of each feature contains a JSON representation
    of the control point with the following items:

    
        row, col:
            row (or line) and col (or pixel) coordinates.
        x, y, z:
            x, y, and z spatial coordinates.
        crs:
            The coordinate reference system for x, y, and z.
        id:
            A unique (within the dataset) identifier for the control
            point.
        info:
            A brief description of the control point.
    