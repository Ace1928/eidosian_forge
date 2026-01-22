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
def timed_write(filepath, dftype, fsize='double', output_directory='times'):
    """Accepts any file with a dataframe readable by the given dataframe type, and writes it out as a variety of file types"""
    assert fsize in ('single', 'double')
    p.dftype = dftype
    df, duration = timed_read(filepath, dftype)
    for ext in write.keys():
        directory, filename = os.path.split(filepath)
        basename, extension = os.path.splitext(filename)
        fname = output_directory + os.path.sep + basename + '.' + ext
        if os.path.exists(fname):
            print('{:28} (keeping existing)'.format(fname), flush=True)
        else:
            filetype = ext.split('.')[-1]
            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    if filetype == 'parq' and df[c].dtype == 'object':
                        df[c] = df[c].str.encode('utf8')
                    else:
                        df[c] = df[c].astype(str)
            if fsize == 'single':
                for colname in df.columns:
                    if df[colname].dtype == 'float64':
                        df[colname] = df[colname].astype(np.float32)
            code = write[ext].get(dftype, None)
            if code is None:
                print('{:28} {:7} Operation not supported'.format(fname, dftype), flush=True)
            else:
                duration, res = code(df, fname, p)
                print('{:28} {:7} {:05.2f}'.format(fname, dftype, duration), flush=True)
            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    df[c] = df[c].astype('category')