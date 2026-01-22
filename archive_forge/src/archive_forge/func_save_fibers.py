import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def save_fibers(oldhdr, oldfib, fname, indices):
    """Stores a new trackvis file fname using only given indices"""
    hdrnew = oldhdr.copy()
    outstreams = []
    for i in indices:
        outstreams.append(oldfib[i])
    n_fib_out = len(outstreams)
    hdrnew['n_count'] = n_fib_out
    iflogger.info('Writing final non-orphan fibers as %s', fname)
    nb.trackvis.write(fname, outstreams, hdrnew)
    return n_fib_out