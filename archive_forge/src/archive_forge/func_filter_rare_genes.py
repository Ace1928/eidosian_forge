from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def filter_rare_genes(data, *extra_data, cutoff=0, min_cells=5):
    """Filter all genes with negligible counts in all but a few cells.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    cutoff : float, optional (default: 0)
        Number of counts above which expression is deemed non-negligible
    min_cells : int, optional (default: 5)
        Minimum number of cells above `cutoff` in order to retain a gene

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    extra_data : array-like, shape=[any, m_features]
        Filtered extra data, if passed.
    """
    gene_sums = measure.gene_capture_count(data, cutoff=cutoff)
    keep_genes_idx = gene_sums >= min_cells
    data = select.select_cols(data, *extra_data, idx=keep_genes_idx)
    return data