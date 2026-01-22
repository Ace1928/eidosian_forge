from . import hdf5
from .utils import _matrix_to_data_frame
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import shutil
import tempfile
import urllib
import warnings
import zipfile
Load HDF5 10X data produced from the 10X Cellranger pipeline.

    Equivalent to `load_10X` but for HDF5 format.

    Parameters
    ----------
    filename: string
        path to HDF5 input data
    genome : str or None, optional (default: None)
        Name of the genome to which CellRanger ran analysis. If None, selects
        the first available genome, and prints all available genomes if more
        than one is available. Invalid for Cellranger 3.0 HDF5 files.
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.
    allow_duplicates : bool, optional (default: None)
        Whether or not to allow duplicate gene names. If None, duplicates are
        allowed for dense input but not for sparse input.
    backend : string, {'tables', 'h5py' or None} optional, default: None
        Selects the HDF5 backend. By default, selects whichever is available,
        using tables if both are available.

    Returns
    -------
    data: array-like, shape=[n_samples, n_features]
        If sparse, data will be a pd.DataFrame[pd.SparseArray]. Otherwise, data will
        be a pd.DataFrame.
    