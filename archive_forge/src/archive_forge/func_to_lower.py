import json
import os
import click
from rasterio.errors import FileOverwriteError
def to_lower(ctx, param, value):
    """Click callback, converts values to lowercase."""
    return value.lower()