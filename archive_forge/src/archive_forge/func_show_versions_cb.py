import rasterio. But if you are using rasterio, you may profit from
import itertools
import logging
import sys
from click_plugins import with_plugins
import click
import cligj
from . import options
import rasterio
from rasterio.session import AWSSession
def show_versions_cb(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    rasterio.show_versions()
    ctx.exit()