import logging
from math import ceil, floor
import sys
import click
import rasterio
from rasterio.crs import CRS
from rasterio.env import setenv
from rasterio.errors import CRSError
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
from rasterio.rio.options import _cb_key_val
from rasterio.transform import Affine
from rasterio.warp import (

    Warp a raster dataset.

    If a template raster is provided using the --like option, the
    coordinate reference system, affine transform, and dimensions of
    that raster will be used for the output.  In this case --dst-crs,
    --bounds, --res, and --dimensions options are not applicable and
    an exception will be raised.

    
        $ rio warp input.tif output.tif --like template.tif

    The output coordinate reference system may be either a PROJ.4 or
    EPSG:nnnn string,

    
        --dst-crs EPSG:4326
        --dst-crs '+proj=longlat +ellps=WGS84 +datum=WGS84'

    or a JSON text-encoded PROJ.4 object.

    
        --dst-crs '{"proj": "utm", "zone": 18, ...}'

    If --dimensions are provided, --res and --bounds are not applicable and an
    exception will be raised.
    Resolution is calculated based on the relationship between the
    raster bounds in the target coordinate system and the dimensions,
    and may produce rectangular rather than square pixels.

    
        $ rio warp input.tif output.tif --dimensions 100 200 \
        > --dst-crs EPSG:4326

    If --bounds are provided, --res is required if --dst-crs is provided
    (defaults to source raster resolution otherwise).

    
        $ rio warp input.tif output.tif \
        > --bounds -78 22 -76 24 --res 0.1 --dst-crs EPSG:4326

    