from __future__ import annotations
import time
import os, os.path, sys, glob, argparse, resource, multiprocessing
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import feather
import fastparquet as fp
from datashader.utils import export_image
from datashader import transfer_functions as tf
from dask import distributed
def timed_agg(df, filepath, plot_width=int(900), plot_height=int(900 * 7.0 / 12), cache_ranges=True):
    global CACHED_RANGES
    start = time.time()
    cvs = ds.Canvas(plot_width, plot_height, x_range=CACHED_RANGES[0], y_range=CACHED_RANGES[1])
    agg = cvs.points(df, p.x, p.y)
    end = time.time()
    if cache_ranges:
        CACHED_RANGES = (cvs.x_range, cvs.y_range)
    img = export_image(tf.shade(agg), filepath, export_path='.')
    return (img, end - start)