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
@hdf5.with_HDF5
def load_10X_HDF5(filename, genome=None, sparse=True, gene_labels='symbol', allow_duplicates=None, backend=None):
    """Load HDF5 10X data produced from the 10X Cellranger pipeline.

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
    """
    if gene_labels not in ['id', 'symbol', 'both']:
        raise ValueError("gene_labels='{}' not recognized. Choose from ['symbol', 'id', 'both']".format(gene_labels))
    if allow_duplicates is None:
        allow_duplicates = not sparse
    with hdf5.open_file(filename, 'r', backend=backend) as f:
        groups = hdf5.list_nodes(f)
        try:
            group = hdf5.get_node(f, 'matrix')
            if genome is not None:
                raise NotImplementedError('Selecting genomes for Cellranger 3.0 files is not currently supported. Please file an issue at https://github.com/KrishnaswamyLab/scprep/issues')
        except (AttributeError, KeyError):
            if genome is None:
                print_genomes = ', '.join(groups)
                genome = groups[0]
                if len(groups) > 1:
                    print('Available genomes: {}. Selecting {} by default'.format(print_genomes, genome))
            try:
                group = hdf5.get_node(f, genome)
            except (AttributeError, KeyError):
                print_genomes = ', '.join(groups)
                raise ValueError('Genome {} not found in {}. Available genomes: {}'.format(genome, filename, print_genomes))
        try:
            features = hdf5.get_node(group, 'features')
            gene_symbols = hdf5.get_node(features, 'name')
            gene_ids = hdf5.get_node(features, 'id')
        except (KeyError, IndexError):
            gene_symbols = hdf5.get_node(group, 'gene_names')
            gene_ids = hdf5.get_node(group, 'genes')
        gene_names = _parse_10x_genes(symbols=[g.decode() for g in hdf5.get_values(gene_symbols)], ids=[g.decode() for g in hdf5.get_values(gene_ids)], gene_labels=gene_labels, allow_duplicates=allow_duplicates)
        cell_names = [b.decode() for b in hdf5.get_values(hdf5.get_node(group, 'barcodes'))]
        data = hdf5.get_values(hdf5.get_node(group, 'data'))
        indices = hdf5.get_values(hdf5.get_node(group, 'indices'))
        indptr = hdf5.get_values(hdf5.get_node(group, 'indptr'))
        shape = hdf5.get_values(hdf5.get_node(group, 'shape'))
        data = sp.csc_matrix((data, indices, indptr), shape=shape)
        data = _matrix_to_data_frame(data.T, gene_names=gene_names, cell_names=cell_names, sparse=sparse)
        return data