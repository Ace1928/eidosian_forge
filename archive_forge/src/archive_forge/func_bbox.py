from __future__ import division
import logging
import click
import cligj
import rasterio
from rasterio.rio import options
from rasterio.features import dataset_features
from rasterio.rio.helpers import write_features
@property
def bbox(self):
    minxs, minys, maxxs, maxys = zip(*self.bboxes)
    return (min(minxs), min(minys), max(maxxs), max(maxys))