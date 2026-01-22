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
def timed_read(filepath, dftype):
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]
    filetype = extension.split('.')[-1]
    code = read[extension].get(dftype, None)
    if code is None:
        return (None, -1)
    p.columns = [p.x] + [p.y] + p.categories
    duration, df = code(filepath, p, filetype)
    return (df, duration)