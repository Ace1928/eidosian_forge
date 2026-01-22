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
def files_handler(ctx, param, value):
    """Process and validate input file names"""
    return value