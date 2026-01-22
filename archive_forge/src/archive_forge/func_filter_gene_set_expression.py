from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def filter_gene_set_expression(data, *extra_data, genes=None, starts_with=None, ends_with=None, exact_word=None, regex=None, cutoff=None, percentile=None, library_size_normalize=False, keep_cells=None, return_expression=False, sample_labels=None, filter_per_sample=None):
    """Remove cells with total expression of a gene set above or below a threshold.

    It is recommended to use :func:`~scprep.plot.plot_gene_set_expression` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    genes : list-like, optional (default: None)
        Integer column indices or string gene names included in gene set
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    exact_word : str, list-like or None, optional (default: None)
        If not None, select genes that contain this exact word.
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression
    cutoff : float or tuple of floats, optional (default: None)
        Expression value above or below which to remove cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain a cell.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above or below the cutoff. If None, defaults to
        'below' for one cutoff and 'between' for two.
    return_expression : bool, optional (default: False)
        If True, also return the values corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_expression : list-like, shape=[m_samples]
        Gene set expression corresponding to retained samples,
        returned only if return_expression is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if keep_cells is None:
        if isinstance(cutoff, numbers.Number) or isinstance(percentile, numbers.Number):
            keep_cells = 'below'
    cell_sums = measure.gene_set_expression(data, genes=genes, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex, library_size_normalize=library_size_normalize)
    return filter_values(data, *extra_data, values=cell_sums, cutoff=cutoff, percentile=percentile, keep_cells=keep_cells, return_values=return_expression, sample_labels=sample_labels, filter_per_sample=filter_per_sample)