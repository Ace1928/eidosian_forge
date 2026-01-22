import json
import logging
from math import ceil
import os
from affine import Affine
import click
import cligj
import rasterio
from rasterio.errors import CRSError
from rasterio.coords import disjoint_bounds
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
import rasterio.shutil
def feature_value(feature):
    if prop and 'properties' in feature:
        return feature['properties'].get(prop, default_value)
    return default_value